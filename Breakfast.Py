import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

def main():
    # Ensure the API key is available in the environment
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    # Initialize the ChatOpenAI model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Maintain conversation history
    messages = [
        HumanMessage(
            content=(
                "You are a healthy breakfast assistant. "
                "Whenever I ask for ideas, provide 5 healthy breakfast ideas, "
                "numbered 1 to 5, with one idea per line, and no extra filler text. "
                "Keep track of everything you suggest so you never repeat ideas."
            )
        )
    ]

    print("Healthy Breakfast Assistant is ready! Type 'exit' or 'quit' to stop.")
    print("Example: 'Give me 5 healthy breakfast ideas'\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Add user message to history
        messages.append(HumanMessage(content=user_input))

        print("AI: ", end="", flush=True)

        response_chunks = []
        # Stream the response chunk by chunk
        for chunk in llm.stream(messages):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                response_chunks.append(chunk.content)
        print()

        # Save the full assistant response to history
        full_response = "".join(response_chunks)
        messages.append(AIMessage(content=full_response))

if __name__ == "__main__":
    main()
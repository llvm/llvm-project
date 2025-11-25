#!/usr/bin/env python3
"""
Enhanced AI CLI - Simple command-line interface for the Enhanced AI Engine

Usage:
    python3 enhanced_ai_cli.py                    # Interactive mode
    python3 enhanced_ai_cli.py "your question"    # Single query mode
"""

import sys
import json
from pathlib import Path
from enhanced_ai_engine import EnhancedAIEngine


class EnhancedAICLI:
    """Simple CLI interface for Enhanced AI Engine"""

    def __init__(self):
        """Initialize CLI"""
        self.engine = None
        self.models = ["fast", "code", "quality_code", "uncensored_code", "large"]
        self.current_model = "uncensored_code"

    def start(self):
        """Start the CLI"""
        self._print_banner()
        self._initialize_engine()

        # Check if single query mode
        if len(sys.argv) > 1:
            query = " ".join(sys.argv[1:])
            self._single_query(query)
        else:
            self._interactive_mode()

    def _print_banner(self):
        """Print welcome banner"""
        print("=" * 70)
        print(" Enhanced AI Engine - CLI Interface")
        print("=" * 70)
        print("\nFeatures:")
        print("  ✅ Conversation history & cross-session memory")
        print("  ✅ Vector embeddings & semantic RAG")
        print("  ✅ Multi-tier response caching")
        print("  ✅ Hierarchical memory (working/short-term/long-term)")
        print("  ✅ Autonomous self-improvement")
        print("  ✅ DSMIL deep integration")
        print("  ✅ 100K-131K token context windows")
        print()

    def _initialize_engine(self):
        """Initialize the AI engine"""
        print("🚀 Initializing Enhanced AI Engine...\n")
        self.engine = EnhancedAIEngine(
            user_id="cli_user",
            enable_self_improvement=True,
            enable_dsmil_integration=True,
            enable_ram_context=True
        )

        # Start a conversation
        self.engine.start_conversation(title="CLI Session")
        print()

    def _single_query(self, query: str):
        """Handle single query and exit"""
        print(f"💬 Query: {query}\n")

        response = self.engine.query(
            prompt=query,
            model=self.current_model,
            use_rag=True,
            use_cache=True
        )

        self._print_response(response)
        self.engine.shutdown()

    def _interactive_mode(self):
        """Interactive mode with command loop"""
        print("Type your questions or use commands:")
        print("  /model <name>   - Switch model (fast|code|quality_code|uncensored_code|large)")
        print("  /stats          - Show statistics")
        print("  /history        - Show conversation history")
        print("  /last           - Show last conversation")
        print("  /search <query> - Search conversations")
        print("  /help           - Show this help")
        print("  /quit           - Exit")
        print()

        try:
            while True:
                # Get user input
                try:
                    user_input = input("\n💬 You: ").strip()
                except EOFError:
                    break

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    if not self._handle_command(user_input):
                        break
                    continue

                # Handle query
                response = self.engine.query(
                    prompt=user_input,
                    model=self.current_model,
                    use_rag=True,
                    use_cache=True
                )

                self._print_response(response)

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted")

        finally:
            self.engine.shutdown()

    def _handle_command(self, command: str) -> bool:
        """
        Handle CLI commands

        Returns:
            False to exit, True to continue
        """
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()

        if cmd == "/quit" or cmd == "/exit" or cmd == "/q":
            print("👋 Goodbye!")
            return False

        elif cmd == "/help" or cmd == "/h":
            self._show_help()

        elif cmd == "/model" or cmd == "/m":
            if len(parts) < 2:
                print(f"Current model: {self.current_model}")
                print(f"Available models: {', '.join(self.models)}")
            else:
                model = parts[1].strip()
                if model in self.models:
                    self.current_model = model
                    print(f"✅ Switched to model: {model}")
                else:
                    print(f"❌ Unknown model: {model}")
                    print(f"   Available: {', '.join(self.models)}")

        elif cmd == "/stats" or cmd == "/s":
            self._show_stats()

        elif cmd == "/history":
            self._show_history()

        elif cmd == "/last":
            self._show_last_conversation()

        elif cmd == "/search":
            if len(parts) < 2:
                print("❌ Usage: /search <query>")
            else:
                self._search_conversations(parts[1])

        else:
            print(f"❌ Unknown command: {cmd}")
            print("   Type /help for available commands")

        return True

    def _show_help(self):
        """Show help message"""
        print("\n📚 Available Commands:")
        print("  /model <name>   - Switch model")
        print("                    (fast|code|quality_code|uncensored_code|large)")
        print("  /stats          - Show system statistics")
        print("  /history        - Show current conversation history")
        print("  /last           - Show last conversation")
        print("  /search <query> - Search across all conversations")
        print("  /help           - Show this help")
        print("  /quit           - Exit")

    def _show_stats(self):
        """Show system statistics"""
        print("\n📊 System Statistics:")
        stats = self.engine.get_statistics()
        print(json.dumps(stats, indent=2))

    def _show_history(self):
        """Show current conversation history"""
        if not self.engine.current_conversation:
            print("❌ No active conversation")
            return

        messages = self.engine.conversation_manager.get_messages(
            self.engine.current_conversation.id
        )

        print(f"\n📝 Conversation History ({len(messages)} messages):")
        for msg in messages:
            role_emoji = "👤" if msg.role == "user" else "🤖"
            content_preview = msg.content[:100]
            if len(msg.content) > 100:
                content_preview += "..."
            print(f"\n{role_emoji} {msg.role.upper()}: {content_preview}")
            if msg.model:
                print(f"   Model: {msg.model}")
            if msg.tokens_output:
                print(f"   Tokens: {msg.tokens_output}")
            if msg.latency_ms:
                print(f"   Latency: {msg.latency_ms}ms")

    def _show_last_conversation(self):
        """Show last conversation"""
        last_conv = self.engine.get_last_conversation()

        if not last_conv:
            print("❌ No previous conversations found")
            return

        print(f"\n📝 Last Conversation:")
        print(f"   ID: {last_conv.id}")
        print(f"   Title: {last_conv.title}")
        print(f"   Started: {last_conv.created_at}")
        print(f"   Updated: {last_conv.updated_at}")

        # Get messages
        messages = self.engine.conversation_manager.get_messages(last_conv.id)
        print(f"\n   Messages: {len(messages)}")

        # Show first few messages
        for i, msg in enumerate(messages[:5]):
            role_emoji = "👤" if msg.role == "user" else "🤖"
            content_preview = msg.content[:80]
            if len(msg.content) > 80:
                content_preview += "..."
            print(f"\n   {role_emoji} {msg.role}: {content_preview}")

        if len(messages) > 5:
            print(f"\n   ... and {len(messages) - 5} more messages")

    def _search_conversations(self, query: str):
        """Search conversations"""
        print(f"\n🔍 Searching for: {query}")

        results = self.engine.search_conversations(query, limit=10)

        if not results:
            print("❌ No conversations found")
            return

        print(f"\n✅ Found {len(results)} conversation(s):")
        for conv in results:
            print(f"\n📝 {conv.title}")
            print(f"   ID: {conv.id}")
            print(f"   Date: {conv.created_at}")

    def _print_response(self, response):
        """Print formatted response"""
        print(f"\n🤖 AI ({response.model}):")
        print(f"{response.content}\n")

        # Show metadata
        metadata_parts = []

        if response.cached:
            metadata_parts.append("⚡ CACHED")

        metadata_parts.append(f"⏱️  {response.latency_ms}ms")
        metadata_parts.append(f"📊 {response.tokens_input}→{response.tokens_output} tokens")
        metadata_parts.append(f"💾 {response.memory_tier}")

        if response.rag_sources:
            metadata_parts.append(f"🔍 {len(response.rag_sources)} RAG sources")

        if response.dsmil_attestation:
            metadata_parts.append(f"🔐 TPM attested")

        print("   " + " | ".join(metadata_parts))

        if response.improvements_suggested:
            print(f"\n💡 Suggestions:")
            for suggestion in response.improvements_suggested:
                print(f"   • {suggestion}")


def main():
    """Main entry point"""
    cli = EnhancedAICLI()
    cli.start()


if __name__ == "__main__":
    main()

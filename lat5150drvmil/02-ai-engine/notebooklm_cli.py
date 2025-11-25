#!/usr/bin/env python3
"""
NotebookLM Natural Language CLI
--------------------------------
Interact with NotebookLM using natural language commands.

Usage:
  python3 notebooklm_cli.py "add this document: /path/to/file.pdf"
  python3 notebooklm_cli.py "summarize my sources"
  python3 notebooklm_cli.py "create a FAQ from all documents"
  python3 notebooklm_cli.py "what are the main findings in my research?"
  python3 notebooklm_cli.py "create a study guide"
  python3 notebooklm_cli.py "synthesize all sources"
  python3 notebooklm_cli.py "list my sources"

The CLI understands natural language and routes to appropriate NotebookLM actions.

Author: LAT5150DRVMIL AI Platform
Version: 1.0.0
"""

import sys
import os
import json
import re
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sub_agents.notebooklm_wrapper import NotebookLMAgent


class NotebookLMCLI:
    """Natural language CLI for NotebookLM"""

    def __init__(self):
        self.agent = NotebookLMAgent()

        # Natural language patterns
        self.patterns = {
            # Add source patterns
            "add_source": [
                r"add (?:this |the )?(?:document|file|source|paper)[:;]?\s*(.+)",
                r"ingest (?:this |the )?(?:document|file|source)[:;]?\s*(.+)",
                r"load (?:this |the )?(?:document|file|source)[:;]?\s*(.+)",
                r"upload (?:this |the )?(?:document|file|source)[:;]?\s*(.+)",
            ],

            # Summarize patterns
            "summarize": [
                r"summarize(?: all)?(?: my)?(?: the)? (?:sources|documents|papers)?",
                r"create (?:a )?summary",
                r"(?:give|show) me (?:a )?summary",
                r"what(?:'s| is) the summary",
            ],

            # FAQ patterns
            "faq": [
                r"create (?:an? )?faq",
                r"generate (?:an? )?faq",
                r"(?:make|build) (?:an? )?faq",
                r"frequently asked questions?",
            ],

            # Study guide patterns
            "study_guide": [
                r"create (?:a )?study guide",
                r"generate (?:a )?study guide",
                r"(?:make|build) (?:a )?study guide",
                r"study (?:guide|material|notes)",
            ],

            # Synthesis patterns
            "synthesis": [
                r"synthesize(?: all)?(?: the)?(?: my)? (?:sources|documents|findings)?",
                r"compare (?:the )?sources",
                r"(?:find|show|what are)(?: the)? (?:connections|relationships)(?: between)?(?: the)? sources",
                r"cross-reference (?:the )?sources",
            ],

            # Briefing patterns
            "briefing": [
                r"create (?:an? )?(?:executive )?briefing",
                r"generate (?:an? )?(?:executive )?briefing",
                r"executive summary",
            ],

            # Query patterns (catch-all for questions)
            "query": [
                r"what (?:are|is|does|did|can|should|would).+",
                r"how (?:does|do|can|should|would|did).+",
                r"why (?:does|do|is|are|did|would).+",
                r"when (?:does|do|is|are|did|should|would).+",
                r"where (?:does|do|is|are|can).+",
                r"who (?:is|are|does|do|can|should).+",
                r"(?:tell|show|explain)(?: me)? (?:about|how|what|why).+",
                r"(?:find|search)(?: for)?.+",
            ],

            # List sources patterns
            "list_sources": [
                r"list(?: my| all)?(?: the)? sources?",
                r"show(?: me)?(?: my| all)?(?: the)? sources?",
                r"what sources (?:do )?(?:i )?have",
                r"what documents (?:have )?(?:i )?(?:added|uploaded|ingested)",
            ],

            # Clear sources patterns
            "clear": [
                r"clear(?: all)?(?: my)? sources?",
                r"delete(?: all)?(?: my)? sources?",
                r"remove(?: all)?(?: my)? sources?",
            ],

            # Status patterns
            "status": [
                r"status",
                r"info(?:rmation)?",
                r"stats?",
                r"how many sources",
            ],
        }

    def parse_command(self, command: str) -> tuple:
        """
        Parse natural language command to action and parameters

        Returns:
            (action, params) tuple
        """
        command_lower = command.lower().strip()

        # Check each pattern category
        for action, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, command_lower)
                if match:
                    # Extract parameters if any
                    params = {}

                    if action == "add_source" and match.groups():
                        # Extract file path or content
                        content = match.group(1).strip()
                        # Check if it's a file path
                        if os.path.exists(content):
                            params["file_path"] = content
                        else:
                            # Try to find file path in original command
                            path_match = re.search(r'["\']?([~/\w\-./]+\.\w+)["\']?', command)
                            if path_match:
                                potential_path = os.path.expanduser(path_match.group(1))
                                if os.path.exists(potential_path):
                                    params["file_path"] = potential_path
                                else:
                                    params["content"] = content
                            else:
                                params["content"] = content

                    elif action == "query":
                        # Use original command as query
                        params["prompt"] = command

                    return action, params

        # Default: treat as query
        return "query", {"prompt": command}

    def execute(self, command: str) -> dict:
        """Execute natural language command"""
        if not self.agent.is_available():
            return {
                "success": False,
                "error": "NotebookLM not available. Please set GOOGLE_API_KEY environment variable."
            }

        # Parse command
        action, params = self.parse_command(command)

        # Execute action
        try:
            if action == "add_source":
                return self.agent.add_source(**params)

            elif action == "summarize":
                return self.agent.summarize_sources()

            elif action == "faq":
                return self.agent.create_faq()

            elif action == "study_guide":
                return self.agent.create_study_guide()

            elif action == "synthesis":
                return self.agent.synthesize()

            elif action == "briefing":
                return self.agent.create_briefing()

            elif action == "query":
                return self.agent.query(**params)

            elif action == "list_sources":
                return self.agent.list_sources()

            elif action == "clear":
                return self.agent.clear_all_sources()

            elif action == "status":
                return self.agent.get_status()

            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}"
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def format_response(self, result: dict) -> str:
        """Format response for display"""
        if not result.get("success"):
            return f"❌ Error: {result.get('error', 'Unknown error')}"

        # Format based on result type
        if "response" in result:
            # Query/summarize/FAQ/etc response
            output = f"✅ {result.get('mode', 'Response').upper()}\n\n"
            output += result["response"]
            output += f"\n\n📚 Sources used: {result.get('sources_used', 0)}"
            if result.get('source_titles'):
                output += f"\n   {', '.join(result['source_titles'])}"
            return output

        elif "sources" in result:
            # List sources response
            sources = result["sources"]
            if not sources:
                return "📚 No sources available. Add sources with: add document /path/to/file"

            output = f"📚 {len(sources)} source(s):\n\n"
            for i, src in enumerate(sources, 1):
                output += f"{i}. {src['title']}\n"
                output += f"   ID: {src['source_id'][:16]}...\n"
                output += f"   Type: {src['source_type']}\n"
                output += f"   Size: {src['content_length']:,} chars\n"
                output += f"   Added: {src['added_at'][:10]}\n\n"
            return output

        elif "source_id" in result:
            # Add source response
            return f"✅ Source added: {result.get('title', 'Untitled')}\n" \
                   f"   ID: {result['source_id']}\n" \
                   f"   Size: {result.get('content_length', 0):,} chars"

        elif "available" in result:
            # Status response
            output = "📊 NotebookLM Status\n\n"
            output += f"Available: {'✅ Yes' if result['available'] else '❌ No'}\n"
            output += f"Sources: {result.get('sources_count', 0)}\n"
            output += f"Notebooks: {result.get('notebooks_count', 0)}\n"
            output += f"Total content: {result.get('total_content_length', 0):,} chars\n"
            output += f"Model: {result.get('model', 'N/A')}\n\n"
            output += "Capabilities:\n"
            for cap in result.get('capabilities', []):
                output += f"  • {cap}\n"
            return output

        elif "message" in result:
            return f"✅ {result['message']}"

        else:
            return json.dumps(result, indent=2)


def main():
    """Main CLI entry point"""
    cli = NotebookLMCLI()

    if len(sys.argv) < 2:
        print("""
╔════════════════════════════════════════════════════════════════╗
║            NotebookLM Natural Language CLI                      ║
║              Powered by Google Gemini 2.0 Flash                 ║
╚════════════════════════════════════════════════════════════════╝

Usage:
  notebooklm_cli.py "your natural language command"

Examples:

  📄 Add Sources:
    "add document /path/to/research.pdf"
    "load file /path/to/notes.txt"
    "ingest source /path/to/specs.md"

  ❓ Ask Questions:
    "what are the main findings?"
    "how does the system work?"
    "summarize the key points"

  📝 Generate Content:
    "create a FAQ"
    "generate a study guide"
    "create an executive briefing"
    "summarize all sources"
    "synthesize my documents"

  📚 Manage Sources:
    "list my sources"
    "show all documents"
    "status"

Natural Language Understanding:
  The CLI understands conversational commands and routes them to
  the appropriate NotebookLM action automatically.

Configuration:
  Set GOOGLE_API_KEY environment variable to use NotebookLM features.
  Example: export GOOGLE_API_KEY="your-api-key-here"
""")
        sys.exit(0)

    # Get command from arguments
    command = " ".join(sys.argv[1:])

    # Execute and display
    result = cli.execute(command)
    output = cli.format_response(result)
    print(output)


if __name__ == "__main__":
    main()

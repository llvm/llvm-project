#!/usr/bin/env python3
"""
Quick AI Query CLI - Ergonomic interface for fast queries

Usage:
  python3 ai_query.py "What is 2+2?"
  python3 ai_query.py --fast "Quick question"
  python3 ai_query.py --code "Write a Python function"
  python3 ai_query.py --rag "Search my docs about TPM"
  echo "What is Docker?" | python3 ai_query.py --stdin

Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import sys
import os
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dsmil_ai_engine import DSMILAIEngine


def print_response(result, verbose=False):
    """Print response in a clean format"""
    if 'response' in result:
        print(result['response'])

        if verbose:
            print()
            print(f"─" * 70)
            print(f"Model: {result.get('model', 'Unknown')} ({result.get('model_tier', 'Unknown')})")
            print(f"Time: {result.get('inference_time', 0):.2f}s")
            print(f"Tokens/sec: {result.get('tokens_per_sec', 0)}")

            if 'attestation' in result:
                att = result['attestation']
                print(f"DSMIL Device: {att.get('dsmil_device', 'Unknown')}")
                print(f"Verified: {'✓' if att.get('verified') else '✗'}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
        if 'suggestion' in result:
            print(f"Suggestion: {result['suggestion']}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="DSMIL AI Engine - Quick Query Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ai_query.py "What is 2+2?"                    # Auto-select model
  ai_query.py --fast "Quick question"           # Use fast model
  ai_query.py --code "Write a sort function"    # Use code model
  ai_query.py --large "Analyze this algorithm"  # Use large model
  ai_query.py --rag "What is TPM?"              # RAG-augmented query
  echo "Hello" | ai_query.py --stdin            # From stdin
  ai_query.py --interactive                     # Interactive mode

Keyboard shortcuts in interactive mode:
  /q or /quit  - Exit
  /h or /help  - Show help
  /fast        - Switch to fast model
  /code        - Switch to code model
  /large       - Switch to large model
  /status      - Show system status
  /rag <file>  - Add file to RAG
        """
    )

    parser.add_argument('query', nargs='*', help='Your query text')
    parser.add_argument('--fast', action='store_true', help='Use fast model')
    parser.add_argument('--code', action='store_true', help='Use code model')
    parser.add_argument('--quality', action='store_true', help='Use quality code model')
    parser.add_argument('--uncensored', action='store_true', help='Use uncensored code model')
    parser.add_argument('--large', action='store_true', help='Use large model')
    parser.add_argument('--rag', action='store_true', help='Enable RAG augmentation (always on by default)')
    parser.add_argument('--stdin', action='store_true', help='Read query from stdin')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')

    args = parser.parse_args()

    # Initialize engine
    engine = DSMILAIEngine()

    # Determine model selection
    model = "auto"
    if args.fast:
        model = "fast"
    elif args.code:
        model = "code"
    elif args.quality:
        model = "quality_code"
    elif args.uncensored:
        model = "uncensored_code"
    elif args.large:
        model = "large"

    # Interactive mode
    if args.interactive:
        print("DSMIL AI Engine - Interactive Mode")
        print("Type your queries or use commands: /q /h /fast /code /large /status /rag")
        print("─" * 70)
        print()

        current_model = "auto"

        while True:
            try:
                query = input(f"[{current_model}]> ").strip()

                if not query:
                    continue

                # Handle commands
                if query.startswith('/'):
                    cmd_parts = query.split(maxsplit=1)
                    cmd = cmd_parts[0].lower()

                    if cmd in ['/q', '/quit', '/exit']:
                        print("Goodbye!")
                        break
                    elif cmd in ['/h', '/help']:
                        print("Commands: /q /quit /fast /code /large /status /rag <file>")
                        continue
                    elif cmd == '/fast':
                        current_model = "fast"
                        print(f"Switched to fast model")
                        continue
                    elif cmd == '/code':
                        current_model = "code"
                        print(f"Switched to code model")
                        continue
                    elif cmd == '/quality':
                        current_model = "quality_code"
                        print(f"Switched to quality code model")
                        continue
                    elif cmd == '/uncensored':
                        current_model = "uncensored_code"
                        print(f"Switched to uncensored code model")
                        continue
                    elif cmd == '/large':
                        current_model = "large"
                        print(f"Switched to large model")
                        continue
                    elif cmd == '/status':
                        status = engine.get_status()
                        print(f"Ollama: {'Connected' if status['ollama']['connected'] else 'Disconnected'}")
                        print(f"Models available: {sum(1 for m in status['models'].values() if m['available'])}")
                        if status.get('rag', {}).get('enabled'):
                            print(f"RAG: {status['rag']['documents']} documents indexed")
                        continue
                    elif cmd == '/rag' and len(cmd_parts) > 1:
                        filepath = cmd_parts[1]
                        print(f"Adding {filepath} to RAG...")
                        result = engine.rag_add_file(filepath)
                        if 'error' in result:
                            print(f"Error: {result['error']}")
                        else:
                            print(f"✓ Added: {result.get('tokens', 0)} tokens")
                        continue
                    else:
                        print(f"Unknown command: {cmd}")
                        continue

                # Process query
                result = engine.generate(query, model_selection=current_model)
                print()
                print_response(result, verbose=True)
                print()

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break

        return

    # Get query from stdin or args
    if args.stdin:
        query = sys.stdin.read().strip()
    else:
        query = ' '.join(args.query).strip()

    if not query:
        parser.print_help()
        sys.exit(1)

    # Generate response
    result = engine.generate(query, model_selection=model)
    print_response(result, verbose=args.verbose)


if __name__ == "__main__":
    main()

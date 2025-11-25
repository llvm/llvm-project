#!/usr/bin/env python3
"""
ai - Clean CLI for DSMIL AI Engine
Simple, fast, beautiful

Usage:
  ai "your question"                    # Quick query
  ai -f "fast question"                 # Fast model
  ai -c "write me some code"            # Code model
  ai -i                                 # Interactive mode
  ai --pydantic "question"              # Type-safe Pydantic mode
  echo "question" | ai                  # Pipe input

Author: DSMIL Integration Framework
Version: 2.1.0 (Pydantic Support)
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from dsmil_ai_engine import DSMILAIEngine, PYDANTIC_AVAILABLE

if PYDANTIC_AVAILABLE:
    from pydantic_models import DSMILQueryRequest, DSMILQueryResult, ModelTier


class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    GRAY = '\033[90m'


def c(code):
    """Color helper"""
    return code if sys.stdout.isatty() else ''


def query(text, model='fast', verbose=False, use_pydantic=False):
    """Run query and display result"""
    # Create engine in appropriate mode
    engine = DSMILAIEngine(pydantic_mode=use_pydantic)

    if verbose:
        print(f"{c(Colors.DIM)}Thinking...{c(Colors.RESET)}", file=sys.stderr)

    # Pydantic mode: Use type-safe models
    if use_pydantic and PYDANTIC_AVAILABLE:
        try:
            # Map model string to ModelTier enum
            model_map = {
                'fast': ModelTier.FAST,
                'code': ModelTier.CODE,
                'quality': ModelTier.QUALITY_CODE,
                'quality_code': ModelTier.QUALITY_CODE,
                'uncensored': ModelTier.UNCENSORED_CODE,
                'uncensored_code': ModelTier.UNCENSORED_CODE,
                'large': ModelTier.LARGE,
            }
            model_tier = model_map.get(model, ModelTier.FAST)

            # Create type-safe request
            request = DSMILQueryRequest(prompt=text, model=model_tier)
            result = engine.generate(request)

            # Result is DSMILQueryResult Pydantic model
            print(result.response)

            if verbose:
                print(f"\n{c(Colors.DIM)}Model: {result.model_used}{c(Colors.RESET)}", file=sys.stderr)
                print(f"{c(Colors.DIM)}Latency: {result.latency_ms:.2f}ms{c(Colors.RESET)}", file=sys.stderr)
                print(f"{c(Colors.DIM)}Confidence: {result.confidence:.2f}{c(Colors.RESET)}", file=sys.stderr)

            return 0
        except Exception as e:
            print(f"{c(Colors.RED)}Error: {str(e)}{c(Colors.RESET)}", file=sys.stderr)
            return 1

    # Legacy mode: Use dict-based responses
    else:
        result = engine.generate(text, model_selection=model)

        if result.get('success'):
            print(result.get('response', ''))
            return 0
        else:
            print(f"{c(Colors.RED)}Error: {result.get('error', 'Unknown error')}{c(Colors.RESET)}", file=sys.stderr)
            return 1


def interactive(use_pydantic=False):
    """Interactive mode"""
    engine = DSMILAIEngine(pydantic_mode=use_pydantic)
    model = 'fast'

    mode_label = "Type-Safe" if use_pydantic and PYDANTIC_AVAILABLE else "Standard"
    print(f"{c(Colors.BOLD)}DSMIL AI{c(Colors.RESET)} {c(Colors.DIM)}Interactive Mode ({mode_label}){c(Colors.RESET)}")
    print(f"{c(Colors.GRAY)}Commands: /fast /code /quality /uncensored /large /quit{c(Colors.RESET)}\n")

    while True:
        try:
            query_text = input(f"{c(Colors.BLUE)}?{c(Colors.RESET)} ").strip()

            if not query_text:
                continue

            # Handle commands
            if query_text.startswith('/'):
                cmd = query_text.lower()
                if cmd in ['/quit', '/q', '/exit']:
                    break
                elif cmd == '/fast':
                    model = 'fast'
                    print(f"{c(Colors.GREEN)}→ Fast model{c(Colors.RESET)}")
                elif cmd == '/code':
                    model = 'code'
                    print(f"{c(Colors.GREEN)}→ Code model{c(Colors.RESET)}")
                elif cmd == '/quality':
                    model = 'quality_code'
                    print(f"{c(Colors.GREEN)}→ Quality model{c(Colors.RESET)}")
                elif cmd == '/uncensored':
                    model = 'uncensored_code'
                    print(f"{c(Colors.GREEN)}→ Uncensored model{c(Colors.RESET)}")
                elif cmd == '/large':
                    model = 'large'
                    print(f"{c(Colors.GREEN)}→ Large model{c(Colors.RESET)}")
                else:
                    print(f"{c(Colors.RED)}Unknown command{c(Colors.RESET)}")
                continue

            # Run query - Pydantic mode
            if use_pydantic and PYDANTIC_AVAILABLE:
                try:
                    model_map = {
                        'fast': ModelTier.FAST,
                        'code': ModelTier.CODE,
                        'quality_code': ModelTier.QUALITY_CODE,
                        'uncensored_code': ModelTier.UNCENSORED_CODE,
                        'large': ModelTier.LARGE,
                    }
                    model_tier = model_map.get(model, ModelTier.FAST)

                    request = DSMILQueryRequest(prompt=query_text, model=model_tier)
                    result = engine.generate(request)

                    print(f"\n{result.response}\n")
                    print(f"{c(Colors.DIM)}[{result.model_used} · {result.latency_ms:.0f}ms · conf: {result.confidence:.2f}]{c(Colors.RESET)}\n")

                except Exception as e:
                    print(f"{c(Colors.RED)}Error: {str(e)}{c(Colors.RESET)}\n")

            # Legacy mode
            else:
                result = engine.generate(query_text, model_selection=model)

                if result.get('success'):
                    print(f"\n{result.get('response', '')}\n")
                else:
                    print(f"{c(Colors.RED)}Error: {result.get('error')}{c(Colors.RESET)}\n")

        except KeyboardInterrupt:
            print()
            break
        except EOFError:
            break

    print(f"{c(Colors.DIM)}Goodbye{c(Colors.RESET)}")


def main():
    """Main entry point"""
    args = sys.argv[1:]

    # No args, check stdin
    if not args:
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()
            if text:
                return query(text)

        # No stdin, show help
        print(f"{c(Colors.BOLD)}ai{c(Colors.RESET)} - DSMIL AI Engine CLI")
        print(f"\n{c(Colors.DIM)}Usage:{c(Colors.RESET)}")
        print(f"  ai \"your question\"")
        print(f"  ai -f \"fast question\"         {c(Colors.GRAY)}# Fast model{c(Colors.RESET)}")
        print(f"  ai -c \"write me code\"         {c(Colors.GRAY)}# Code model{c(Colors.RESET)}")
        print(f"  ai -q \"complex question\"      {c(Colors.GRAY)}# Quality model{c(Colors.RESET)}")
        print(f"  ai -i                         {c(Colors.GRAY)}# Interactive{c(Colors.RESET)}")
        print(f"  ai --pydantic \"question\"      {c(Colors.GRAY)}# Type-safe mode{c(Colors.RESET)}")
        print(f"  echo \"question\" | ai          {c(Colors.GRAY)}# Pipe input{c(Colors.RESET)}")
        print()
        pydantic_status = f"{c(Colors.GREEN)}Available" if PYDANTIC_AVAILABLE else f"{c(Colors.YELLOW)}Not installed"
        print(f"{c(Colors.DIM)}Pydantic AI: {pydantic_status}{c(Colors.RESET)}")
        print()
        return 1

    # Parse args
    model = 'fast'
    interactive_mode = False
    verbose = False
    use_pydantic = False
    text_parts = []

    i = 0
    while i < len(args):
        arg = args[i]

        if arg in ['-f', '--fast']:
            model = 'fast'
        elif arg in ['-c', '--code']:
            model = 'code'
        elif arg in ['-q', '--quality']:
            model = 'quality_code'
        elif arg in ['-u', '--uncensored']:
            model = 'uncensored_code'
        elif arg in ['-l', '--large']:
            model = 'large'
        elif arg in ['-i', '--interactive']:
            interactive_mode = True
        elif arg in ['-v', '--verbose']:
            verbose = True
        elif arg in ['-p', '--pydantic', '--type-safe']:
            if not PYDANTIC_AVAILABLE:
                print(f"{c(Colors.YELLOW)}⚠  Pydantic mode requested but not available{c(Colors.RESET)}", file=sys.stderr)
                print(f"{c(Colors.DIM)}Install with: pip install pydantic pydantic-ai{c(Colors.RESET)}", file=sys.stderr)
                return 1
            use_pydantic = True
        elif arg in ['-h', '--help']:
            # Show help (same as no args)
            return main.__doc__
        else:
            text_parts.append(arg)

        i += 1

    # Interactive mode
    if interactive_mode:
        interactive(use_pydantic=use_pydantic)
        return 0

    # Query mode
    text = ' '.join(text_parts).strip()
    if text:
        return query(text, model, verbose, use_pydantic)
    else:
        print(f"{c(Colors.RED)}Error: No query provided{c(Colors.RESET)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{c(Colors.DIM)}Interrupted{c(Colors.RESET)}", file=sys.stderr)
        sys.exit(130)

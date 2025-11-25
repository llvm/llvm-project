#!/usr/bin/env python3
"""
AI Keyboard Interface - Entry Point

Lightning-fast keyboard-first interface for power users.
Based on HumanLayer's "superhuman speed" workflows.

Usage:
    python3 ai_keyboard.py
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_orchestrator import UnifiedAIOrchestrator
from keyboard_interface import KeyboardInterface


def main():
    """Main entry point"""
    try:
        # Initialize orchestrator
        print("🚀 Initializing DSMIL AI Orchestrator...\n")
        orchestrator = UnifiedAIOrchestrator(enable_ace=True)

        # Create keyboard interface
        interface = KeyboardInterface(orchestrator)

        # Run interface
        interface.run()

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

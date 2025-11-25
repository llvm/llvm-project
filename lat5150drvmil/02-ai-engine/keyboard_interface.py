#!/usr/bin/env python3
"""
Keyboard-First Command Interface - "Superhuman Speed" for Power Users

Lightning-fast command interface inspired by HumanLayer's keyboard-first workflows.
Designed for builders who value speed and control.

Key Features:
- Single-key shortcuts for common operations
- Command palette for advanced operations
- No mouse required
- Context-aware commands
- History and autocomplete
- Vim-like modal interface

Based on: HumanLayer/CodeLayer keyboard-first design philosophy
"""

import os
import sys
import readline
import atexit
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import json


class CommandMode(Enum):
    """Command modes (vim-like)"""
    NORMAL = "normal"      # Default mode - single-key commands
    COMMAND = "command"    # Command palette mode
    WORKFLOW = "workflow"  # Workflow execution mode
    PARALLEL = "parallel"  # Parallel task mode


@dataclass
class Command:
    """Represents a keyboard command"""
    key: str              # Keyboard shortcut
    name: str            # Command name
    description: str     # Description
    handler: Callable    # Handler function
    mode: CommandMode = CommandMode.NORMAL
    requires_input: bool = False
    aliases: List[str] = None

    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


class KeyboardInterface:
    """
    Keyboard-First Command Interface

    Fast, keyboard-driven interface for AI agent orchestration.
    Inspired by HumanLayer's "superhuman speed" workflows.
    """

    def __init__(self, orchestrator):
        """
        Initialize keyboard interface

        Args:
            orchestrator: UnifiedAIOrchestrator instance
        """
        self.orchestrator = orchestrator
        self.mode = CommandMode.NORMAL
        self.commands: Dict[str, Command] = {}
        self.command_history: List[str] = []
        self.running = False

        # Initialize readline for command history
        self._setup_readline()

        # Register built-in commands
        self._register_commands()

        print("\n⚡ Keyboard-First Interface - Superhuman Speed")
        print("   Type '?' for help, 'q' to quit\n")

    def _setup_readline(self):
        """Setup readline for command history and autocomplete"""
        # History file
        history_file = os.path.expanduser("~/.dsmil_keyboard_history")

        # Load history
        try:
            readline.read_history_file(history_file)
        except FileNotFoundError:
            pass

        # Save history on exit
        atexit.register(readline.write_history_file, history_file)

        # Set history length
        readline.set_history_length(1000)

        # Tab completion
        readline.parse_and_bind("tab: complete")
        readline.set_completer(self._completer)

    def _completer(self, text: str, state: int):
        """Autocomplete for commands"""
        options = [
            cmd.name for cmd in self.commands.values()
            if cmd.name.startswith(text)
        ]
        if state < len(options):
            return options[state]
        return None

    def _register_commands(self):
        """Register built-in keyboard commands"""

        # === QUERY COMMANDS ===
        self.register_command(Command(
            key="q",
            name="query",
            description="Quick AI query",
            handler=self._cmd_query,
            requires_input=True
        ))

        # === WORKFLOW COMMANDS ===
        self.register_command(Command(
            key="w",
            name="workflow",
            description="Start ACE-FCA workflow",
            handler=self._cmd_workflow,
            requires_input=True
        ))

        self.register_command(Command(
            key="W",
            name="workflow-interactive",
            description="Interactive workflow setup",
            handler=self._cmd_workflow_interactive
        ))

        # === PARALLEL EXECUTION ===
        self.register_command(Command(
            key="p",
            name="parallel",
            description="Enter parallel mode (multi-agent)",
            handler=self._cmd_parallel_mode
        ))

        self.register_command(Command(
            key="P",
            name="parallel-status",
            description="Show parallel execution status",
            handler=self._cmd_parallel_status
        ))

        # === SUBAGENT COMMANDS ===
        self.register_command(Command(
            key="r",
            name="research",
            description="Research subagent (file search)",
            handler=self._cmd_research,
            requires_input=True
        ))

        self.register_command(Command(
            key="s",
            name="summarize",
            description="Summarize content",
            handler=self._cmd_summarize,
            requires_input=True
        ))

        # === WORKTREE COMMANDS ===
        self.register_command(Command(
            key="t",
            name="worktree-create",
            description="Create new worktree",
            handler=self._cmd_worktree_create,
            requires_input=True
        ))

        self.register_command(Command(
            key="T",
            name="worktree-list",
            description="List all worktrees",
            handler=self._cmd_worktree_list
        ))

        # === CONTEXT COMMANDS ===
        self.register_command(Command(
            key="c",
            name="context",
            description="Show context stats",
            handler=self._cmd_context_stats
        ))

        self.register_command(Command(
            key="C",
            name="compact",
            description="Compact context",
            handler=self._cmd_compact_context
        ))

        # === STATUS COMMANDS ===
        self.register_command(Command(
            key="i",
            name="status",
            description="System status",
            handler=self._cmd_status,
            aliases=["info"]
        ))

        # === HELP & NAVIGATION ===
        self.register_command(Command(
            key="?",
            name="help",
            description="Show help",
            handler=self._cmd_help,
            aliases=["h"]
        ))

        self.register_command(Command(
            key=":",
            name="command-palette",
            description="Open command palette",
            handler=self._cmd_palette
        ))

        self.register_command(Command(
            key="ESC",
            name="escape",
            description="Return to normal mode",
            handler=self._cmd_escape
        ))

    def register_command(self, command: Command):
        """Register a keyboard command"""
        self.commands[command.key] = command

        # Register aliases
        for alias in command.aliases:
            self.commands[alias] = command

    def _cmd_query(self, input_text: str = None):
        """Quick AI query command"""
        query = input_text or input("🔍 Query: ").strip()
        if not query:
            return

        print("⏳ Thinking...")
        result = self.orchestrator.query(query)

        print(f"\n{result.get('response', 'No response')}\n")

        # Show metadata
        print(f"⚡ {result.get('backend', 'unknown')} | "
              f"{result.get('model', 'unknown')} | "
              f"{result.get('total_time', 0):.1f}s")

    def _cmd_workflow(self, input_text: str = None):
        """Start ACE-FCA workflow"""
        task = input_text or input("📋 Task description: ").strip()
        if not task:
            return

        print(f"\n🔄 Starting workflow: {task}")
        print("   Phases: Research → Plan → Implement → Verify\n")

        if not hasattr(self.orchestrator, 'execute_workflow'):
            print("❌ ACE-FCA workflows not available")
            return

        result = self.orchestrator.execute_workflow(
            task_description=task,
            task_type="feature",
            complexity="medium"
        )

        if result.get('success'):
            print(f"\n✅ Workflow complete!")
            print(f"   Phases: {', '.join(result.get('phases_completed', []))}")
        else:
            print(f"\n❌ Workflow failed: {result.get('error', 'Unknown')}")

    def _cmd_workflow_interactive(self):
        """Interactive workflow setup"""
        print("\n📋 ACE-FCA Workflow Setup")

        task = input("   Task: ").strip()
        if not task:
            return

        print("\n   Type: (f)eature, (b)ugfix, (r)efactor, (a)nalysis")
        type_choice = input("   → ").strip().lower()
        task_type = {
            'f': 'feature', 'b': 'bugfix',
            'r': 'refactor', 'a': 'analysis'
        }.get(type_choice, 'feature')

        print("\n   Complexity: (s)imple, (m)edium, (c)omplex")
        complexity_choice = input("   → ").strip().lower()
        complexity = {
            's': 'simple', 'm': 'medium', 'c': 'complex'
        }.get(complexity_choice, 'medium')

        print("\n   Constraints (comma-separated, optional):")
        constraints_input = input("   → ").strip()
        constraints = [c.strip() for c in constraints_input.split(',') if c.strip()]

        print(f"\n▶️  Starting: {task} [{task_type}, {complexity}]")

        result = self.orchestrator.execute_workflow(
            task_description=task,
            task_type=task_type,
            complexity=complexity,
            constraints=constraints
        )

        if result.get('success'):
            print(f"\n✅ Complete! Phases: {', '.join(result['phases_completed'])}")
        else:
            print(f"\n❌ Failed: {result.get('error')}")

    def _cmd_parallel_mode(self):
        """Enter parallel execution mode"""
        print("\n🚀 Parallel Execution Mode")
        print("   Submit multiple tasks, they'll run concurrently\n")

        self.mode = CommandMode.PARALLEL

        print("Commands:")
        print("  w <task>  - Submit workflow")
        print("  q <query> - Submit query")
        print("  r <query> - Submit research")
        print("  s         - Show status")
        print("  x         - Execute all")
        print("  ESC       - Exit parallel mode\n")

        # TODO: Implement parallel mode loop
        print("⚠️  Parallel mode implementation in progress")
        self.mode = CommandMode.NORMAL

    def _cmd_parallel_status(self):
        """Show parallel execution status"""
        if not hasattr(self.orchestrator, 'parallel_executor'):
            print("❌ Parallel executor not available")
            return

        status = self.orchestrator.parallel_executor.get_status()

        print(f"\n🚀 Parallel Executor Status:")
        print(f"   Running: {status['currently_running']}/{status['max_concurrent_agents']}")
        print(f"   Queue: {status['queue_size']} pending")
        print(f"   Total: {status['total_tasks']} tasks")
        print(f"   Completed: {status['tasks_by_status']['completed']}")
        print(f"   Failed: {status['tasks_by_status']['failed']}")

    def _cmd_research(self, input_text: str = None):
        """Research subagent command"""
        query = input_text or input("🔍 Research query: ").strip()
        if not query:
            return

        print("⏳ Researching...")

        result = self.orchestrator.use_subagent('research', {
            'query': query,
            'search_paths': ['.'],
            'file_patterns': ['*.py', '*.js', '*.ts', '*.md']
        })

        if result.get('success'):
            print(f"\n📄 Research Results:")
            print(result.get('compressed_output', 'No output'))
            print(f"\n📊 Files analyzed: {result.get('metadata', {}).get('files_found', 0)}")
        else:
            print(f"❌ Error: {result.get('error')}")

    def _cmd_summarize(self, input_text: str = None):
        """Summarize content command"""
        if input_text:
            content = input_text
        else:
            print("📝 Enter content to summarize (Ctrl+D when done):")
            lines = []
            try:
                while True:
                    line = input()
                    lines.append(line)
            except EOFError:
                pass
            content = '\n'.join(lines)

        if not content:
            return

        print("\n⏳ Summarizing...")

        result = self.orchestrator.use_subagent('summarizer', {
            'content': content,
            'max_tokens': 200
        })

        if result.get('success'):
            print(f"\n📄 Summary:")
            print(result.get('compressed_output', 'No output'))
        else:
            print(f"❌ Error: {result.get('error')}")

    def _cmd_worktree_create(self, input_text: str = None):
        """Create worktree command"""
        branch = input_text or input("🌿 Branch name: ").strip()

        if not hasattr(self.orchestrator, 'worktree_manager'):
            print("❌ Worktree manager not available")
            return

        worktree = self.orchestrator.worktree_manager.create_worktree(
            branch_name=branch,
            description=f"Created via keyboard interface"
        )

        if worktree:
            print(f"✅ Worktree created: {worktree.path}")
        else:
            print("❌ Failed to create worktree")

    def _cmd_worktree_list(self):
        """List worktrees command"""
        if not hasattr(self.orchestrator, 'worktree_manager'):
            print("❌ Worktree manager not available")
            return

        worktrees = self.orchestrator.worktree_manager.list_worktrees()

        if not worktrees:
            print("📋 No worktrees")
            return

        print(f"\n🌿 Worktrees ({len(worktrees)}):")
        for wt in worktrees:
            print(f"   • {wt.branch}")
            print(f"     Path: {wt.path}")
            if wt.task_id:
                print(f"     Task: {wt.task_id}")
            print()

    def _cmd_context_stats(self):
        """Show context stats"""
        stats = self.orchestrator.get_context_stats()

        if 'error' in stats:
            print(f"❌ {stats['error']}")
            return

        print(f"\n📊 Context Stats:")
        print(f"   Tokens: {stats['total_tokens']}/{stats['max_tokens']} ({stats['utilization_percent']})")
        print(f"   Optimal range: {stats['in_optimal_range']} ✓" if stats['in_optimal_range'] else "   ⚠️  Outside optimal range")
        print(f"   Compactions: {stats['compaction_count']}")
        print(f"   Current phase: {stats.get('current_phase', 'None')}")

    def _cmd_compact_context(self):
        """Compact context"""
        print("⏳ Compacting context...")

        result = self.orchestrator.compact_context()

        if 'error' in result:
            print(f"❌ {result['error']}")
            return

        if result.get('compacted'):
            print(f"✅ Context compacted")
            print(f"   Freed: {result['tokens_freed']} tokens")
            print(f"   Blocks removed: {result['blocks_removed']}")
            print(f"   New utilization: {result['utilization']:.1%}")
        else:
            print(f"ℹ️  {result.get('reason', 'No compaction needed')}")

    def _cmd_status(self):
        """System status"""
        status = self.orchestrator.get_status()

        print(f"\n⚙️  System Status:")

        # Backends
        backends = status.get('backends', {})
        for name, info in backends.items():
            available = "✓" if info.get('available') else "✗"
            print(f"   {available} {name}: {info.get('priority', 'N/A')}")

        # ACE-FCA
        if 'ace_fca' in status:
            ace = status['ace_fca']
            if ace.get('available'):
                print(f"\n   ✓ ACE-FCA: Enabled")
                ctx = ace.get('context_stats', {})
                if ctx:
                    print(f"     Context: {ctx.get('utilization_percent', '0%')}")

    def _cmd_help(self):
        """Show help"""
        print("\n⚡ Keyboard-First Commands:")
        print()

        # Group by category
        categories = {
            "Query": ['q', 'r', 's'],
            "Workflow": ['w', 'W'],
            "Parallel": ['p', 'P'],
            "Worktree": ['t', 'T'],
            "Context": ['c', 'C'],
            "System": ['i', '?', ':']
        }

        for category, keys in categories.items():
            print(f"  {category}:")
            for key in keys:
                if key in self.commands:
                    cmd = self.commands[key]
                    print(f"    {key:3} - {cmd.description}")
            print()

    def _cmd_palette(self):
        """Command palette"""
        print("\n⌘ Command Palette")
        print("   Type command name or part of it\n")

        query = input("   > ").strip().lower()

        # Find matching commands
        matches = [
            cmd for cmd in self.commands.values()
            if query in cmd.name.lower() or query in cmd.description.lower()
        ]

        if not matches:
            print("   No matches")
            return

        if len(matches) == 1:
            # Execute directly
            self._execute_command(matches[0])
        else:
            # Show matches
            print("\n   Matches:")
            for i, cmd in enumerate(matches[:10], 1):
                print(f"   {i}. {cmd.name} - {cmd.description}")

            choice = input("\n   Select (1-{}): ".format(len(matches))).strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(matches):
                    self._execute_command(matches[idx])
            except ValueError:
                pass

    def _cmd_escape(self):
        """Escape to normal mode"""
        self.mode = CommandMode.NORMAL
        print("⬅️  Normal mode")

    def _execute_command(self, command: Command):
        """Execute a command"""
        try:
            if command.requires_input:
                input_text = input(f"{command.name} > ").strip()
                command.handler(input_text)
            else:
                command.handler()
        except Exception as e:
            print(f"❌ Error: {e}")

    def run(self):
        """Run keyboard interface loop"""
        self.running = True

        while self.running:
            try:
                # Show mode indicator
                mode_indicator = {
                    CommandMode.NORMAL: "⚡",
                    CommandMode.COMMAND: "⌘",
                    CommandMode.WORKFLOW: "🔄",
                    CommandMode.PARALLEL: "🚀"
                }.get(self.mode, "?")

                # Get input
                key = input(f"{mode_indicator} ").strip()

                # Handle quit
                if key.lower() in ['quit', 'exit', 'q!']:
                    print("\n👋 Goodbye!\n")
                    break

                # Execute command
                if key in self.commands:
                    self._execute_command(self.commands[key])
                elif ' ' in key:
                    # Command with argument (e.g., "q what is python")
                    cmd_key, arg = key.split(' ', 1)
                    if cmd_key in self.commands:
                        cmd = self.commands[cmd_key]
                        if cmd.requires_input:
                            cmd.handler(arg)
                        else:
                            print(f"⚠️  Command '{cmd_key}' doesn't take arguments")
                    else:
                        print(f"❌ Unknown command: {cmd_key}")
                else:
                    print(f"❌ Unknown command: {key} (type ? for help)")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                break
            except EOFError:
                print("\n\n👋 Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


# Main entry point
if __name__ == "__main__":
    print("Keyboard-First Interface requires UnifiedAIOrchestrator")
    print("Run via: python3 ai_keyboard.py")

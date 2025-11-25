#!/usr/bin/env python3
"""
DSMIL AI Engine - Modern Clean Interface
Streamlined, elegant, and powerful AI management

Author: DSMIL Integration Framework
Version: 2.0.0 (Modern UI)
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from dsmil_ai_engine import DSMILAIEngine

# ACE-FCA imports
try:
    from unified_orchestrator import UnifiedAIOrchestrator
    ACE_AVAILABLE = True
except ImportError:
    ACE_AVAILABLE = False


class Colors:
    """ANSI color codes for modern terminal UI"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Clean, professional colors
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'


class ModernAITUI:
    """Modern, clean AI Engine interface"""

    def __init__(self):
        self.engine = DSMILAIEngine()
        self.colors_enabled = sys.stdout.isatty()

        # Initialize orchestrator for ACE-FCA workflows
        if ACE_AVAILABLE:
            self.orchestrator = UnifiedAIOrchestrator(enable_ace=True)
        else:
            self.orchestrator = None

    def c(self, color_code):
        """Return color code if colors enabled, else empty string"""
        return color_code if self.colors_enabled else ''

    def clear(self):
        """Clear screen"""
        os.system('clear' if os.name != 'nt' else 'cls')

    def header(self, title, subtitle=None):
        """Modern clean header"""
        self.clear()
        print(f"\n{self.c(Colors.BOLD)}{self.c(Colors.CYAN)}DSMIL AI{self.c(Colors.RESET)} {self.c(Colors.DIM)}→ {title}{self.c(Colors.RESET)}")
        if subtitle:
            print(f"{self.c(Colors.GRAY)}{subtitle}{self.c(Colors.RESET)}")
        print()

    def status_line(self):
        """Clean status line with essential info"""
        status = self.engine.get_status()

        # Ollama status
        ollama = "●" if status['ollama']['connected'] else "○"
        ollama_color = Colors.GREEN if status['ollama']['connected'] else Colors.RED

        # Models count
        models_count = sum(1 for m in status['models'].values() if m['available'])
        models_color = Colors.GREEN if models_count >= 3 else Colors.YELLOW

        # RAG status
        rag_docs = status.get('rag', {}).get('documents', 0)
        rag_color = Colors.GREEN if rag_docs > 0 else Colors.GRAY

        print(f"{self.c(Colors.DIM)}Status  {self.c(ollama_color)}{ollama}{self.c(Colors.RESET)} Ollama  "
              f"{self.c(models_color)}{models_count}/5{self.c(Colors.RESET)} models  "
              f"{self.c(rag_color)}{rag_docs}{self.c(Colors.RESET)} docs{self.c(Colors.RESET)}\n")

    def menu(self, options, prompt="Choose"):
        """Clean menu display"""
        for key, label in options:
            print(f"  {self.c(Colors.BLUE)}{key}{self.c(Colors.RESET)}  {label}")

        print()
        choice = input(f"{self.c(Colors.DIM)}{prompt} → {self.c(Colors.RESET)}").strip().lower()
        return choice

    def info(self, text):
        """Info message"""
        print(f"{self.c(Colors.CYAN)}→{self.c(Colors.RESET)} {text}")

    def success(self, text):
        """Success message"""
        print(f"{self.c(Colors.GREEN)}✓{self.c(Colors.RESET)} {text}")

    def error(self, text):
        """Error message"""
        print(f"{self.c(Colors.RED)}✗{self.c(Colors.RESET)} {text}")

    def wait(self):
        """Wait for user"""
        input(f"\n{self.c(Colors.DIM)}Press Enter to continue{self.c(Colors.RESET)}")

    # ===== MAIN MENU =====

    def main_menu(self):
        """Main menu - clean and focused"""
        while True:
            self.header("Main Menu", "Hardware-attested AI inference")
            self.status_line()

            menu_options = [
                ('q', 'Query AI'),
                ('m', 'Models'),
                ('r', 'RAG Knowledge'),
                ('g', 'Guardrails'),
                ('s', 'Status'),
            ]

            # Add ACE-FCA workflow option if available
            if ACE_AVAILABLE and self.orchestrator and self.orchestrator.ace_enabled:
                menu_options.insert(1, ('w', 'ACE Workflow (Research→Plan→Implement→Verify)'))

            # Add parallel execution option if available
            if self.orchestrator and hasattr(self.orchestrator, 'parallel_enabled') and self.orchestrator.parallel_enabled:
                menu_options.insert(2, ('p', 'Parallel Execution (M U L T I C L A U D E)'))

            menu_options.append(('x', 'Exit'))

            choice = self.menu(menu_options)

            if choice == 'x':
                print(f"\n{self.c(Colors.DIM)}Goodbye{self.c(Colors.RESET)}\n")
                break
            elif choice == 'q':
                self.query_interface()
            elif choice == 'w' and ACE_AVAILABLE:
                self.ace_workflow_interface()
            elif choice == 'p' and hasattr(self.orchestrator, 'parallel_enabled'):
                self.parallel_interface()
            elif choice == 'm':
                self.models_interface()
            elif choice == 'r':
                self.rag_interface()
            elif choice == 'g':
                self.guardrails_interface()
            elif choice == 's':
                self.status_interface()

    # ===== QUERY INTERFACE =====

    def query_interface(self):
        """Clean query interface"""
        self.header("AI Query", "Ask anything, automatic RAG augmentation")

        # Model selection
        print(f"{self.c(Colors.DIM)}Models{self.c(Colors.RESET)}")
        print(f"  {self.c(Colors.BLUE)}f{self.c(Colors.RESET)} fast (Phi-3)")
        print(f"  {self.c(Colors.BLUE)}c{self.c(Colors.RESET)} code (DeepSeek)")
        print(f"  {self.c(Colors.BLUE)}q{self.c(Colors.RESET)} quality (Llama)")
        print(f"  {self.c(Colors.BLUE)}u{self.c(Colors.RESET)} uncensored (WizardLM)")
        print(f"  {self.c(Colors.BLUE)}l{self.c(Colors.RESET)} large (Qwen2.5)")
        print()

        model_map = {'f': 'fast', 'c': 'code', 'q': 'quality', 'u': 'uncensored', 'l': 'large'}
        model_choice = input(f"{self.c(Colors.DIM)}Model (f/c/q/u/l) → {self.c(Colors.RESET)}").strip().lower()
        model = model_map.get(model_choice, 'fast')

        print()
        query = input(f"{self.c(Colors.BLUE)}?{self.c(Colors.RESET)} ").strip()

        if not query:
            return

        print(f"\n{self.c(Colors.DIM)}Thinking...{self.c(Colors.RESET)}")
        result = self.engine.generate(query, model_selection=model)

        print(f"\n{self.c(Colors.GREEN)}Response{self.c(Colors.RESET)}")
        print(f"{result.get('response', 'No response')}")

        self.wait()

    # ===== MODELS INTERFACE =====

    def models_interface(self):
        """Models management - clean list"""
        self.header("Models", "Available AI models")

        status = self.engine.get_status()
        models = status.get('models', {})

        for key, model in models.items():
            available = model.get('available', False)
            indicator = f"{self.c(Colors.GREEN)}●{self.c(Colors.RESET)}" if available else f"{self.c(Colors.GRAY)}○{self.c(Colors.RESET)}"

            name = model.get('name', 'Unknown')
            size = model.get('size', '')

            print(f"  {indicator} {self.c(Colors.BOLD)}{name}{self.c(Colors.RESET)}")
            if size:
                print(f"     {self.c(Colors.DIM)}{size}{self.c(Colors.RESET)}")
            print()

        self.wait()

    # ===== RAG INTERFACE =====

    def rag_interface(self):
        """RAG knowledge base - simplified"""
        while True:
            self.header("RAG Knowledge", "Retrieval augmented generation")

            stats = self.engine.rag_get_stats()
            if not stats.get('error'):
                docs = stats.get('total_documents', 0)
                tokens = stats.get('total_tokens', 0)
                print(f"{self.c(Colors.DIM)}Knowledge base  {self.c(Colors.CYAN)}{docs}{self.c(Colors.RESET)} documents  "
                      f"{self.c(Colors.CYAN)}{tokens:,}{self.c(Colors.RESET)} tokens\n")

            choice = self.menu([
                ('a', 'Add file'),
                ('d', 'Add directory'),
                ('s', 'Search'),
                ('l', 'List documents'),
                ('b', 'Back')
            ])

            if choice == 'b':
                break
            elif choice == 'a':
                path = input(f"\n{self.c(Colors.DIM)}File path → {self.c(Colors.RESET)}").strip()
                if path:
                    self.info("Indexing...")
                    result = self.engine.rag_add_file(path)
                    if result.get('status') == 'success':
                        self.success(f"Added {path}")
                    else:
                        self.error(result.get('error', 'Failed'))
                    self.wait()
            elif choice == 'd':
                path = input(f"\n{self.c(Colors.DIM)}Directory path → {self.c(Colors.RESET)}").strip()
                if path:
                    self.info("Indexing recursively...")
                    result = self.engine.rag_add_folder(path)
                    self.success(f"Indexed {result.get('files_added', 0)} files")
                    self.wait()
            elif choice == 's':
                query = input(f"\n{self.c(Colors.DIM)}Search → {self.c(Colors.RESET)}").strip()
                if query:
                    result = self.engine.rag_search(query, max_results=5)
                    results = result.get('results', [])
                    print()
                    for doc in results:
                        print(f"{self.c(Colors.BLUE)}•{self.c(Colors.RESET)} {doc.get('filename', 'Unknown')}")
                        preview = doc.get('preview', '')[:100]
                        print(f"  {self.c(Colors.DIM)}{preview}...{self.c(Colors.RESET)}\n")
                    self.wait()
            elif choice == 'l':
                result = self.engine.rag_list_documents()
                docs = result.get('documents', [])
                print()
                for doc in docs:
                    tokens = doc.get('tokens', doc.get('token_count', 0))
                    print(f"{self.c(Colors.BLUE)}•{self.c(Colors.RESET)} {doc.get('filename', 'Unknown')} "
                          f"{self.c(Colors.DIM)}({tokens:,} tokens){self.c(Colors.RESET)}")
                self.wait()

    # ===== ACE-FCA WORKFLOW INTERFACE =====

    def ace_workflow_interface(self):
        """ACE-FCA phase-based workflow interface"""
        if not self.orchestrator or not self.orchestrator.ace_enabled:
            self.error("ACE-FCA not available")
            self.wait()
            return

        self.header("ACE-FCA Workflow", "Research → Plan → Implement → Verify")

        # Show context stats if available
        if self.orchestrator.ace_engine:
            stats = self.orchestrator.get_context_stats()
            if 'error' not in stats:
                util = stats.get('utilization_percent', '0%')
                print(f"{self.c(Colors.DIM)}Context: {self.c(Colors.CYAN)}{util}{self.c(Colors.RESET)} "
                      f"{'✓' if stats.get('in_optimal_range') else '⚠️'}\n")

        # Get task description
        print(f"{self.c(Colors.DIM)}Describe the coding task{self.c(Colors.RESET)}")
        task_desc = input(f"{self.c(Colors.BLUE)}Task → {self.c(Colors.RESET)}").strip()

        if not task_desc:
            return

        print()

        # Get task type
        print(f"{self.c(Colors.DIM)}Task type{self.c(Colors.RESET)}")
        print(f"  {self.c(Colors.BLUE)}f{self.c(Colors.RESET)} feature")
        print(f"  {self.c(Colors.BLUE)}b{self.c(Colors.RESET)} bugfix")
        print(f"  {self.c(Colors.BLUE)}r{self.c(Colors.RESET)} refactor")
        print(f"  {self.c(Colors.BLUE)}a{self.c(Colors.RESET)} analysis")
        print()

        type_map = {'f': 'feature', 'b': 'bugfix', 'r': 'refactor', 'a': 'analysis'}
        type_choice = input(f"{self.c(Colors.DIM)}Type (f/b/r/a) → {self.c(Colors.RESET)}").strip().lower()
        task_type = type_map.get(type_choice, 'feature')

        print()

        # Get complexity
        print(f"{self.c(Colors.DIM)}Complexity{self.c(Colors.RESET)}")
        print(f"  {self.c(Colors.BLUE)}s{self.c(Colors.RESET)} simple")
        print(f"  {self.c(Colors.BLUE)}m{self.c(Colors.RESET)} medium")
        print(f"  {self.c(Colors.BLUE)}c{self.c(Colors.RESET)} complex")
        print()

        complexity_map = {'s': 'simple', 'm': 'medium', 'c': 'complex'}
        complexity_choice = input(f"{self.c(Colors.DIM)}Complexity (s/m/c) → {self.c(Colors.RESET)}").strip().lower()
        complexity = complexity_map.get(complexity_choice, 'medium')

        print()

        # Optional: Get constraints
        print(f"{self.c(Colors.DIM)}Constraints (optional, comma-separated){self.c(Colors.RESET)}")
        constraints_input = input(f"{self.c(Colors.DIM)}Constraints → {self.c(Colors.RESET)}").strip()
        constraints = [c.strip() for c in constraints_input.split(',') if c.strip()] if constraints_input else []

        print()

        # Confirm
        print(f"{self.c(Colors.BOLD)}Summary{self.c(Colors.RESET)}")
        print(f"  Task: {task_desc}")
        print(f"  Type: {task_type}")
        print(f"  Complexity: {complexity}")
        if constraints:
            print(f"  Constraints: {', '.join(constraints)}")
        print()

        confirm = input(f"{self.c(Colors.DIM)}Start workflow? (y/n) → {self.c(Colors.RESET)}").strip().lower()
        if confirm != 'y':
            self.info("Cancelled")
            self.wait()
            return

        print()
        self.info("Starting ACE-FCA workflow...")
        print(f"{self.c(Colors.DIM)}Phases: Research → Plan → Implement → Verify{self.c(Colors.RESET)}")
        print(f"{self.c(Colors.DIM)}Human review checkpoints enabled{self.c(Colors.RESET)}")
        print()

        # Execute workflow
        try:
            result = self.orchestrator.execute_workflow(
                task_description=task_desc,
                task_type=task_type,
                complexity=complexity,
                constraints=constraints,
                model_preference="quality_code"
            )

            if result.get('success'):
                print()
                self.success("Workflow completed successfully!")
                print()

                phases = result.get('phases_completed', [])
                print(f"{self.c(Colors.BOLD)}Phases completed:{self.c(Colors.RESET)} {', '.join(phases)}")

                # Show review checkpoints
                reviews = result.get('review_checkpoints', [])
                if reviews:
                    approved = sum(1 for r in reviews if r.get('approved'))
                    print(f"{self.c(Colors.BOLD)}Review checkpoints:{self.c(Colors.RESET)} {approved}/{len(reviews)} approved")

                # Show context stats
                ctx_stats = result.get('context_stats', {})
                if ctx_stats:
                    print(f"{self.c(Colors.BOLD)}Context usage:{self.c(Colors.RESET)} {ctx_stats.get('utilization_percent', 'N/A')}")
                    print(f"{self.c(Colors.BOLD)}Compactions:{self.c(Colors.RESET)} {ctx_stats.get('compaction_count', 0)}")

                print()

                # Offer to show outputs
                show = input(f"{self.c(Colors.DIM)}Show phase outputs? (y/n) → {self.c(Colors.RESET)}").strip().lower()
                if show == 'y':
                    print()
                    if 'research_output' in result:
                        print(f"{self.c(Colors.BOLD)}{self.c(Colors.CYAN)}RESEARCH:{self.c(Colors.RESET)}")
                        print(result['research_output'])
                        print()

                    if 'plan_output' in result:
                        print(f"{self.c(Colors.BOLD)}{self.c(Colors.CYAN)}PLAN:{self.c(Colors.RESET)}")
                        print(result['plan_output'])
                        print()

                    if 'implementation_notes' in result:
                        print(f"{self.c(Colors.BOLD)}{self.c(Colors.CYAN)}IMPLEMENTATION:{self.c(Colors.RESET)}")
                        print(result['implementation_notes'])
                        print()

                    if 'verification_results' in result:
                        print(f"{self.c(Colors.BOLD)}{self.c(Colors.CYAN)}VERIFICATION:{self.c(Colors.RESET)}")
                        print(result['verification_results'])
                        print()

            else:
                print()
                self.error(f"Workflow failed: {result.get('error', 'Unknown error')}")
                if 'phase_stopped' in result:
                    print(f"{self.c(Colors.DIM)}Stopped at: {result['phase_stopped']}{self.c(Colors.RESET)}")
                print()

        except Exception as e:
            self.error(f"Error executing workflow: {str(e)}")

        self.wait()

    # ===== PARALLEL EXECUTION INTERFACE =====

    def parallel_interface(self):
        """Parallel execution interface - M U L T I C L A U D E"""
        if not self.orchestrator or not hasattr(self.orchestrator, 'parallel_executor'):
            self.error("Parallel execution not available")
            self.wait()
            return

        while True:
            self.header("Parallel Execution", "M U L T I C L A U D E - Run multiple agents concurrently")

            # Show status
            status = self.orchestrator.parallel_executor.get_status()

            print(f"{self.c(Colors.DIM)}Executor Status{self.c(Colors.RESET)}")
            print(f"  Running: {self.c(Colors.CYAN)}{status['currently_running']}/{status['max_concurrent_agents']}{self.c(Colors.RESET)}")
            print(f"  Queue: {self.c(Colors.YELLOW)}{status['queue_size']}{self.c(Colors.RESET)} pending")
            print(f"  Completed: {self.c(Colors.GREEN)}{status['tasks_by_status']['completed']}{self.c(Colors.RESET)}")
            if status['tasks_by_status']['failed'] > 0:
                print(f"  Failed: {self.c(Colors.RED)}{status['tasks_by_status']['failed']}{self.c(Colors.RESET)}")
            print()

            choice = self.menu([
                ('w', 'Submit workflow'),
                ('q', 'Submit query'),
                ('r', 'Submit research'),
                ('l', 'List tasks'),
                ('s', 'Start executor'),
                ('x', 'Stop executor'),
                ('b', 'Back')
            ])

            if choice == 'b':
                break
            elif choice == 'w':
                task = input(f"\n{self.c(Colors.DIM)}Workflow task → {self.c(Colors.RESET)}").strip()
                if task:
                    task_id = self.orchestrator.parallel_executor.submit_workflow(task)
                    self.success(f"Submitted: {task_id}")
                    self.wait()
            elif choice == 'q':
                query = input(f"\n{self.c(Colors.DIM)}Query → {self.c(Colors.RESET)}").strip()
                if query:
                    task_id = self.orchestrator.parallel_executor.submit_query(query)
                    self.success(f"Submitted: {task_id}")
                    self.wait()
            elif choice == 'r':
                query = input(f"\n{self.c(Colors.DIM)}Research query → {self.c(Colors.RESET)}").strip()
                if query:
                    task_id = self.orchestrator.parallel_executor.submit_subagent(
                        'research',
                        {'query': query, 'search_paths': ['.'], 'file_patterns': ['*.py']},
                        description=f"Research: {query}"
                    )
                    self.success(f"Submitted: {task_id}")
                    self.wait()
            elif choice == 'l':
                tasks = self.orchestrator.parallel_executor.list_tasks()
                print()
                if not tasks:
                    print(f"  {self.c(Colors.DIM)}No tasks{self.c(Colors.RESET)}")
                else:
                    for task in tasks[:10]:  # Show last 10
                        status_color = {
                            'pending': Colors.YELLOW,
                            'running': Colors.CYAN,
                            'completed': Colors.GREEN,
                            'failed': Colors.RED,
                            'cancelled': Colors.GRAY
                        }.get(task['status'], Colors.WHITE)

                        print(f"  {self.c(status_color)}{task['status']:12}{self.c(Colors.RESET)} {task['task_id']} - {task['description'][:50]}")
                print()
                self.wait()
            elif choice == 's':
                import asyncio
                asyncio.create_task(self.orchestrator.parallel_executor.start())
                self.success("Executor started")
                self.wait()
            elif choice == 'x':
                import asyncio
                asyncio.create_task(self.orchestrator.parallel_executor.stop())
                self.info("Executor stopped")
                self.wait()

    # ===== GUARDRAILS INTERFACE =====

    def guardrails_interface(self):
        """Guardrails configuration"""
        self.header("Guardrails", "Content filtering & safety")

        status = self.engine.get_status()
        guardrails = status.get('guardrails', {})

        print(f"{self.c(Colors.DIM)}Content filtering{self.c(Colors.RESET)}")
        enabled = guardrails.get('enabled', False)
        indicator = f"{self.c(Colors.GREEN)}ON{self.c(Colors.RESET)}" if enabled else f"{self.c(Colors.GRAY)}OFF{self.c(Colors.RESET)}"
        print(f"  Status: {indicator}")
        print()

        blocked = guardrails.get('blocked_topics', [])
        if blocked:
            print(f"{self.c(Colors.DIM)}Blocked topics{self.c(Colors.RESET)}")
            for topic in blocked:
                print(f"  {self.c(Colors.RED)}•{self.c(Colors.RESET)} {topic}")
            print()

        self.wait()

    # ===== STATUS INTERFACE =====

    def status_interface(self):
        """System status - comprehensive"""
        self.header("System Status", "Complete system overview")

        status = self.engine.get_status()

        # Ollama
        ollama = status.get('ollama', {})
        connected = ollama.get('connected', False)
        print(f"{self.c(Colors.BOLD)}Ollama{self.c(Colors.RESET)}")
        print(f"  Status: {self.c(Colors.GREEN if connected else Colors.RED)}{'Connected' if connected else 'Disconnected'}{self.c(Colors.RESET)}")
        if connected and ollama.get('url'):
            print(f"  URL: {self.c(Colors.DIM)}{ollama['url']}{self.c(Colors.RESET)}")
        print()

        # Models
        models = status.get('models', {})
        available_count = sum(1 for m in models.values() if m.get('available'))
        print(f"{self.c(Colors.BOLD)}Models{self.c(Colors.RESET)}")
        print(f"  Available: {self.c(Colors.CYAN)}{available_count}/5{self.c(Colors.RESET)}")
        print()

        # RAG
        rag = status.get('rag', {})
        rag_enabled = rag.get('enabled', False)
        print(f"{self.c(Colors.BOLD)}RAG System{self.c(Colors.RESET)}")
        print(f"  Status: {self.c(Colors.GREEN if rag_enabled else Colors.GRAY)}{'Enabled' if rag_enabled else 'Disabled'}{self.c(Colors.RESET)}")
        if rag_enabled:
            print(f"  Documents: {self.c(Colors.CYAN)}{rag.get('documents', 0)}{self.c(Colors.RESET)}")
        print()

        # DSMIL
        dsmil = status.get('dsmil', {})
        mode5 = dsmil.get('mode5', {})
        print(f"{self.c(Colors.BOLD)}DSMIL Integration{self.c(Colors.RESET)}")
        print(f"  Mode 5 Level: {self.c(Colors.CYAN)}{mode5.get('mode5_level', 'N/A')}{self.c(Colors.RESET)}")
        print()

        self.wait()


def main():
    """Entry point"""
    try:
        tui = ModernAITUI()
        tui.main_menu()
    except KeyboardInterrupt:
        print(f"\n{Colors.DIM}Interrupted{Colors.RESET}\n")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.RESET}\n")


if __name__ == "__main__":
    main()

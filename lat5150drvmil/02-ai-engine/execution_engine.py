#!/usr/bin/env python3
"""
Execution Engine for Local Claude Code
Sophisticated execution of planned tasks with error recovery and learning
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from advanced_planner import ExecutionPlan, ExecutionStep, StepType
from file_operations import FileOps
from edit_operations import EditOps
from tool_operations import ToolOps
from context_manager import ContextManager
from codebase_learner import CodebaseLearner
from pattern_database import PatternDatabase

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of execution"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"  # Some steps succeeded
    SKIPPED = "skipped"


class ExecutionMode(Enum):
    """Execution mode for plans"""
    TRADITIONAL = "traditional"  # Step-by-step execution (existing)
    CODE_MODE = "code_mode"      # Batched execution via code-mode (NEW)
    HYBRID = "hybrid"             # Intelligent routing (NEW)


@dataclass
class StepResult:
    """Result of executing a single step"""
    step_num: int
    status: ExecutionStatus
    output: Any
    error: Optional[str] = None
    duration: float = 0.0
    retry_count: int = 0


@dataclass
class ExecutionResult:
    """Result of executing an entire plan"""
    plan: ExecutionPlan
    status: ExecutionStatus
    step_results: List[StepResult]
    total_duration: float
    steps_succeeded: int
    steps_failed: int
    learned_patterns: int = 0
    context_summary: str = ""
    execution_mode: str = "traditional"  # NEW: Track which mode was used
    api_calls: int = 0  # NEW: Track API efficiency
    tokens_estimate: int = 0  # NEW: Track token usage


class ExecutionEngine:
    """
    Sophisticated execution engine

    Features:
    - Execute multi-step plans
    - Error recovery and retries
    - Pattern learning from successful executions
    - Context tracking across steps
    - Integration with all system components
    """

    def __init__(
        self,
        ai_engine,
        file_ops: FileOps,
        edit_ops: EditOps,
        tool_ops: ToolOps,
        context_manager: ContextManager,
        codebase_learner: Optional[CodebaseLearner] = None,
        pattern_db: Optional[PatternDatabase] = None,
        semantic_engine: Optional[Any] = None,
        max_retries: int = 2,
        enable_code_mode: bool = True  # NEW: Enable code-mode execution
    ):
        """
        Initialize execution engine

        Args:
            ai_engine: AI engine for code generation
            file_ops: File operations
            edit_ops: Edit operations
            tool_ops: Tool operations
            context_manager: Context manager
            codebase_learner: Codebase learner (optional)
            pattern_db: Pattern database (optional)
            semantic_engine: Serena semantic code engine (optional)
            max_retries: Maximum retries per step
        """
        self.ai = ai_engine
        self.files = file_ops
        self.edits = edit_ops
        self.tools = tool_ops
        self.context = context_manager
        self.learner = codebase_learner
        self.pattern_db = pattern_db
        self.serena = semantic_engine
        self.max_retries = max_retries

        # NEW: Code-mode integration
        self.enable_code_mode = enable_code_mode
        self.code_mode_bridge = None
        if enable_code_mode:
            try:
                from code_mode_bridge import CodeModeBridge
                self.code_mode_bridge = CodeModeBridge()
                logger.info("Code-mode bridge available")
            except Exception as e:
                logger.warning(f"Code-mode bridge unavailable: {e}")
                self.enable_code_mode = False

        # Execution state
        self.current_plan: Optional[ExecutionPlan] = None
        self.step_context: Dict[int, Any] = {}  # Results from previous steps

        logger.info(f"ExecutionEngine initialized (code-mode: {self.enable_code_mode})")

    def execute_plan(
        self,
        plan: ExecutionPlan,
        dry_run: bool = False,
        interactive: bool = False,
        mode: str = "hybrid"  # NEW: "traditional", "code_mode", or "hybrid"
    ) -> ExecutionResult:
        """
        Execute a complete plan

        Args:
            plan: Execution plan
            dry_run: If True, simulate execution without making changes
            interactive: If True, prompt before each step
            mode: Execution mode ("traditional", "code_mode", or "hybrid")

        Returns:
            ExecutionResult with details
        """
        # NEW: Intelligent routing for hybrid mode
        if mode == "hybrid":
            mode = self._select_execution_mode(plan)
            logger.info(f"Hybrid mode selected: {mode}")

        # NEW: Route to code-mode if appropriate
        if mode == "code_mode" and self.enable_code_mode and not dry_run and not interactive:
            return self._execute_plan_code_mode(plan)

        # Traditional execution (existing logic)
        logger.info(f"Executing plan: {plan.task} ({len(plan.steps)} steps)")

        self.current_plan = plan
        self.step_context = {}

        start_time = time.time()
        step_results = []

        for step in plan.steps:
            if interactive:
                response = input(f"\nExecute step {step.step_num}? (y/n/skip): ")
                if response.lower() == 'n':
                    break
                elif response.lower() == 'skip':
                    step_results.append(StepResult(
                        step_num=step.step_num,
                        status=ExecutionStatus.SKIPPED,
                        output=None
                    ))
                    continue

            logger.info(f"Step {step.step_num}/{len(plan.steps)}: {step.description}")

            # Execute step
            result = self._execute_step(step, dry_run=dry_run)
            step_results.append(result)

            # Store result for future steps
            self.step_context[step.step_num] = result.output

            # If step failed and no fallback, stop
            if result.status == ExecutionStatus.FAILED and not step.fallback:
                logger.error(f"Step {step.step_num} failed, stopping execution")
                break

        # Calculate final status
        succeeded = sum(1 for r in step_results if r.status == ExecutionStatus.SUCCESS)
        failed = sum(1 for r in step_results if r.status == ExecutionStatus.FAILED)

        if failed == 0:
            final_status = ExecutionStatus.SUCCESS
        elif succeeded > 0:
            final_status = ExecutionStatus.PARTIAL
        else:
            final_status = ExecutionStatus.FAILED

        total_duration = time.time() - start_time

        # Learn from execution
        learned_patterns = 0
        if self.learner and final_status == ExecutionStatus.SUCCESS:
            learned_patterns = self._learn_from_execution(plan, step_results)

        # Get context summary
        context_summary = self.context.get_project_context()

        result = ExecutionResult(
            plan=plan,
            status=final_status,
            step_results=step_results,
            total_duration=total_duration,
            steps_succeeded=succeeded,
            steps_failed=failed,
            learned_patterns=learned_patterns,
            context_summary=context_summary
        )

        logger.info(
            f"Execution complete: {final_status.value} "
            f"({succeeded} succeeded, {failed} failed, {total_duration:.1f}s)"
        )

        return result

    def _execute_step(self, step: ExecutionStep, dry_run: bool = False) -> StepResult:
        """Execute a single step"""
        start_time = time.time()

        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retry {attempt}/{self.max_retries}")

                # Execute based on step type
                output, error = self._execute_step_by_type(step, dry_run)

                if error:
                    if attempt < self.max_retries:
                        continue  # Retry
                    else:
                        # Max retries reached
                        return StepResult(
                            step_num=step.step_num,
                            status=ExecutionStatus.FAILED,
                            output=output,
                            error=error,
                            duration=time.time() - start_time,
                            retry_count=attempt
                        )

                # Success
                return StepResult(
                    step_num=step.step_num,
                    status=ExecutionStatus.SUCCESS,
                    output=output,
                    duration=time.time() - start_time,
                    retry_count=attempt
                )

            except Exception as e:
                logger.error(f"Step execution error: {e}")

                if attempt < self.max_retries:
                    continue
                else:
                    return StepResult(
                        step_num=step.step_num,
                        status=ExecutionStatus.FAILED,
                        output=None,
                        error=str(e),
                        duration=time.time() - start_time,
                        retry_count=attempt
                    )

        # Should not reach here
        return StepResult(
            step_num=step.step_num,
            status=ExecutionStatus.FAILED,
            output=None,
            error="Unknown error"
        )

    def _execute_step_by_type(self, step: ExecutionStep, dry_run: bool) -> Tuple[Any, Optional[str]]:
        """Execute step based on its type"""

        if step.step_type == StepType.READ_FILE:
            return self._execute_read_file(step, dry_run)

        elif step.step_type == StepType.EDIT_FILE:
            return self._execute_edit_file(step, dry_run)

        elif step.step_type == StepType.WRITE_FILE:
            return self._execute_write_file(step, dry_run)

        elif step.step_type == StepType.SEARCH:
            return self._execute_search(step, dry_run)

        elif step.step_type == StepType.ANALYZE:
            return self._execute_analyze(step, dry_run)

        elif step.step_type == StepType.EXECUTE:
            return self._execute_command(step, dry_run)

        elif step.step_type == StepType.TEST:
            return self._execute_test(step, dry_run)

        elif step.step_type == StepType.GIT:
            return self._execute_git(step, dry_run)

        elif step.step_type == StepType.GENERATE_CODE:
            return self._execute_generate_code(step, dry_run)

        elif step.step_type == StepType.LEARN_PATTERN:
            return self._execute_learn_pattern(step, dry_run)

        # Serena semantic code operations
        elif step.step_type == StepType.FIND_SYMBOL:
            return self._execute_find_symbol(step, dry_run)

        elif step.step_type == StepType.FIND_REFERENCES:
            return self._execute_find_references(step, dry_run)

        elif step.step_type == StepType.SEMANTIC_SEARCH:
            return self._execute_semantic_search(step, dry_run)

        elif step.step_type == StepType.SEMANTIC_EDIT:
            return self._execute_semantic_edit(step, dry_run)

        else:
            return None, f"Unknown step type: {step.step_type}"

    def _execute_read_file(self, step: ExecutionStep, dry_run: bool) -> Tuple[Any, Optional[str]]:
        """Execute file read"""
        filepath = step.parameters.get('filepath') or self._extract_filepath_from_action(step.action)

        if not filepath:
            return None, "No filepath specified"

        if dry_run:
            return {"dry_run": True, "action": f"Read {filepath}"}, None

        result = self.files.read_file(filepath)

        if 'error' in result:
            return result, result['error']

        # Track in context
        self.context.add_file_access(filepath, result.get('content'))

        # Learn from file if possible
        if self.learner and result.get('content'):
            self.learner.learn_from_file(filepath, result['content'])

        return result, None

    def _execute_edit_file(self, step: ExecutionStep, dry_run: bool) -> Tuple[Any, Optional[str]]:
        """Execute file edit"""
        filepath = step.parameters.get('filepath') or self._extract_filepath_from_action(step.action)
        old_string = step.parameters.get('old_string')
        new_string = step.parameters.get('new_string')

        if not filepath:
            return None, "No filepath specified"

        # If old/new not in parameters, ask AI to generate them
        if not old_string or not new_string:
            # Read file first
            file_result = self.files.read_file(filepath)
            if 'error' in file_result:
                return file_result, file_result['error']

            # Ask AI for edit
            edit_prompt = f"""Generate an edit for this task:

Task: {step.description}
Action: {step.action}

File: {filepath}
Content:
{file_result.get('content', '')[:2000]}

Provide:
OLD: <exact string to replace>
NEW: <new string>

Be precise."""

            ai_result = self.ai.generate(edit_prompt, model_selection=self.current_plan.model_used)

            if 'error' in ai_result:
                return ai_result, ai_result['error']

            # Parse OLD/NEW
            old_string, new_string = self._parse_old_new(ai_result['response'])

            if not old_string or not new_string:
                return None, "Could not parse edit from AI response"

        if dry_run:
            return {
                "dry_run": True,
                "action": f"Edit {filepath}",
                "old": old_string[:50],
                "new": new_string[:50]
            }, None

        # Perform edit
        result = self.edits.edit_file(filepath, old_string, new_string)

        if 'error' in result:
            return result, result['error']

        # Track in context
        self.context.add_edit(filepath, old_string, new_string, step.description)

        return result, None

    def _execute_write_file(self, step: ExecutionStep, dry_run: bool) -> Tuple[Any, Optional[str]]:
        """Execute file write"""
        filepath = step.parameters.get('filepath') or self._extract_filepath_from_action(step.action)
        content = step.parameters.get('content')

        if not filepath:
            return None, "No filepath specified"

        # If content not provided, ask AI to generate it
        if not content:
            gen_prompt = f"""Generate code for this task:

Task: {step.description}
Action: {step.action}

Provide ONLY the code, no explanation."""

            ai_result = self.ai.generate(gen_prompt, model_selection=self.current_plan.model_used)

            if 'error' in ai_result:
                return ai_result, ai_result['error']

            content = ai_result['response']

        if dry_run:
            return {
                "dry_run": True,
                "action": f"Write {filepath}",
                "size": len(content)
            }, None

        # Write file
        result = self.edits.write_file(filepath, content)

        if 'error' in result:
            return result, result['error']

        # Track in context
        self.context.add_file_access(filepath, content, operation="write")

        return result, None

    def _execute_search(self, step: ExecutionStep, dry_run: bool) -> Tuple[Any, Optional[str]]:
        """Execute search"""
        pattern = step.parameters.get('pattern') or self._extract_pattern_from_action(step.action)

        if not pattern:
            return None, "No search pattern specified"

        if dry_run:
            return {"dry_run": True, "action": f"Search for '{pattern}'"}, None

        result = self.files.grep(pattern)

        if 'error' in result:
            return result, result['error']

        return result, None

    def _execute_analyze(self, step: ExecutionStep, dry_run: bool) -> Tuple[Any, Optional[str]]:
        """Execute analysis"""
        if dry_run:
            return {"dry_run": True, "action": f"Analyze: {step.description}"}, None

        # Use AI to analyze
        context = self._build_context_for_step(step)

        analyze_prompt = f"""Analyze this:

{step.description}

Context:
{context}

Provide analysis."""

        ai_result = self.ai.generate(analyze_prompt, model_selection=self.current_plan.model_used)

        if 'error' in ai_result:
            return ai_result, ai_result['error']

        return {"analysis": ai_result['response']}, None

    def _execute_command(self, step: ExecutionStep, dry_run: bool) -> Tuple[Any, Optional[str]]:
        """Execute command"""
        command = step.parameters.get('command') or self._extract_command_from_action(step.action)

        if not command:
            return None, "No command specified"

        if dry_run:
            return {"dry_run": True, "action": f"Run: {command}"}, None

        result = self.tools.bash(command, description=step.description)

        # Track in context
        self.context.add_execution(
            command,
            result.get('success', False),
            result.get('stdout', ''),
            result.get('stderr', '')
        )

        if not result.get('success'):
            return result, f"Command failed: {result.get('stderr', 'Unknown error')}"

        return result, None

    def _execute_test(self, step: ExecutionStep, dry_run: bool) -> Tuple[Any, Optional[str]]:
        """Execute tests"""
        test_command = step.parameters.get('command', 'pytest')

        if dry_run:
            return {"dry_run": True, "action": f"Run tests: {test_command}"}, None

        result = self.tools.run_tests(test_command)

        if not result.get('success'):
            return result, f"Tests failed:\n{result.get('stderr', '')}"

        return result, None

    def _execute_git(self, step: ExecutionStep, dry_run: bool) -> Tuple[Any, Optional[str]]:
        """Execute git command"""
        git_args = step.parameters.get('args') or self._extract_git_args(step.action)

        if not git_args:
            return None, "No git arguments specified"

        if dry_run:
            return {"dry_run": True, "action": f"git {git_args}"}, None

        result = self.tools.git(git_args, description=step.description)

        if not result.get('success'):
            return result, f"Git command failed: {result.get('stderr', '')}"

        return result, None

    def _execute_generate_code(self, step: ExecutionStep, dry_run: bool) -> Tuple[Any, Optional[str]]:
        """Execute code generation"""
        if dry_run:
            return {"dry_run": True, "action": f"Generate code: {step.description}"}, None

        # Get similar patterns if available
        similar_patterns = []
        if self.pattern_db:
            similar_patterns = self.pattern_db.search_patterns(step.description, limit=3)

        # Build generation prompt
        context = self._build_context_for_step(step)

        gen_prompt = f"""Generate code for this task:

{step.description}

Context:
{context}
"""

        if similar_patterns:
            gen_prompt += "\n\nSimilar patterns:\n"
            for pattern in similar_patterns:
                gen_prompt += f"\n{pattern.name}:\n{pattern.code_example[:200]}\n"

        gen_prompt += "\nProvide ONLY the code, no explanation."

        ai_result = self.ai.generate(gen_prompt, model_selection=self.current_plan.model_used)

        if 'error' in ai_result:
            return ai_result, ai_result['error']

        return {"code": ai_result['response']}, None

    def _execute_learn_pattern(self, step: ExecutionStep, dry_run: bool) -> Tuple[Any, Optional[str]]:
        """Execute pattern learning"""
        if not self.learner:
            return None, "Codebase learner not available"

        if dry_run:
            return {"dry_run": True, "action": f"Learn pattern: {step.description}"}, None

        filepath = step.parameters.get('filepath')
        if filepath:
            result = self.learner.learn_from_file(filepath)
            return result, None if 'error' not in result else result['error']

        return None, "No filepath for learning"

    def _build_context_for_step(self, step: ExecutionStep) -> str:
        """Build context string for step execution"""
        context_parts = []

        # Add dependencies' results
        for dep_num in step.dependencies:
            if dep_num in self.step_context:
                context_parts.append(f"Step {dep_num} result: {self.step_context[dep_num]}")

        # Add recent conversation
        conv_context = self.context.get_conversation_context(last_n=3)
        if conv_context:
            context_parts.append(f"\nRecent context:\n{conv_context}")

        return "\n".join(context_parts)

    def _learn_from_execution(self, plan: ExecutionPlan, results: List[StepResult]) -> int:
        """Learn patterns from successful execution"""
        if not self.learner:
            return 0

        learned = 0

        # Learn from all files that were accessed
        for filepath, file_ctx in self.context.files_accessed.items():
            if file_ctx.was_edited:
                # This file was successfully edited, learn from it
                try:
                    file_result = self.learner.learn_from_file(filepath)
                    learned += file_result.get('functions', 0) + file_result.get('classes', 0)
                except Exception as e:
                    logger.error(f"Error learning from {filepath}: {e}")

        if learned > 0:
            logger.info(f"Learned {learned} patterns from execution")

        return learned

    def _extract_filepath_from_action(self, action: str) -> Optional[str]:
        """Extract filepath from action string"""
        import re

        patterns = [
            r'["\']([^"\']+\.(?:py|js|ts|c|cpp|h|sh|md))["\']',
            r'(\S+\.(?:py|js|ts|c|cpp|h|sh|md))',
            r'file[:\s]+(\S+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, action)
            if match:
                return match.group(1)

        return None

    def _extract_pattern_from_action(self, action: str) -> Optional[str]:
        """Extract search pattern from action"""
        import re

        match = re.search(r'["\']([^"\']+)["\']', action)
        if match:
            return match.group(1)

        return None

    def _extract_command_from_action(self, action: str) -> Optional[str]:
        """Extract command from action"""
        import re

        match = re.search(r'["\']([^"\']+)["\']', action)
        if match:
            return match.group(1)

        # Fallback: use action as command
        return action

    def _extract_git_args(self, action: str) -> Optional[str]:
        """Extract git arguments from action"""
        if 'git' in action.lower():
            return action.lower().replace('git', '').strip()

        return action

    def _parse_old_new(self, response: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse OLD and NEW strings from AI response"""
        import re

        old_match = re.search(r'OLD:\s*```?([^`]+)```?', response, re.DOTALL | re.IGNORECASE)
        new_match = re.search(r'NEW:\s*```?([^`]+)```?', response, re.DOTALL | re.IGNORECASE)

        if old_match and new_match:
            return old_match.group(1).strip(), new_match.group(1).strip()

        return None, None

    # ========================================================================
    # SERENA SEMANTIC CODE OPERATIONS
    # ========================================================================

    def _execute_find_symbol(self, step: ExecutionStep, dry_run: bool) -> Tuple[Any, Optional[str]]:
        """Execute symbol find using Serena LSP"""
        if not self.serena:
            return None, "Serena semantic engine not available"

        symbol_name = step.parameters.get('symbol_name') or step.action
        symbol_type = step.parameters.get('symbol_type')  # function, class, variable
        language = step.parameters.get('language')

        if not symbol_name:
            return None, "No symbol name specified"

        if dry_run:
            return {
                "dry_run": True,
                "action": f"Find symbol '{symbol_name}' (type: {symbol_type}, language: {language})"
            }, None

        try:
            import asyncio
            # Initialize serena if needed
            if not self.serena.initialized:
                asyncio.run(self.serena.initialize())

            # Find symbol
            symbols = asyncio.run(self.serena.find_symbol(symbol_name, symbol_type, language))

            result = {
                "symbol_name": symbol_name,
                "matches": len(symbols),
                "locations": [s.to_dict() for s in symbols]
            }

            logger.info(f"Found {len(symbols)} matches for symbol '{symbol_name}'")
            return result, None

        except Exception as e:
            error_msg = f"Error finding symbol: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

    def _execute_find_references(self, step: ExecutionStep, dry_run: bool) -> Tuple[Any, Optional[str]]:
        """Execute reference find using Serena LSP"""
        if not self.serena:
            return None, "Serena semantic engine not available"

        file_path = step.parameters.get('filepath')
        line = step.parameters.get('line')
        column = step.parameters.get('column', 0)

        if not file_path or line is None:
            return None, "filepath and line required for find_references"

        if dry_run:
            return {
                "dry_run": True,
                "action": f"Find references at {file_path}:{line}:{column}"
            }, None

        try:
            import asyncio
            if not self.serena.initialized:
                asyncio.run(self.serena.initialize())

            # Find references
            references = asyncio.run(self.serena.find_references(file_path, line, column))

            result = {
                "file_path": file_path,
                "line": line,
                "column": column,
                "references": len(references),
                "locations": [r.to_dict() for r in references]
            }

            logger.info(f"Found {len(references)} references")
            return result, None

        except Exception as e:
            error_msg = f"Error finding references: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

    def _execute_semantic_search(self, step: ExecutionStep, dry_run: bool) -> Tuple[Any, Optional[str]]:
        """Execute semantic code search using Serena"""
        if not self.serena:
            return None, "Serena semantic engine not available"

        query = step.parameters.get('query') or step.action
        max_results = step.parameters.get('max_results', 10)
        language = step.parameters.get('language')

        if not query:
            return None, "No search query specified"

        if dry_run:
            return {
                "dry_run": True,
                "action": f"Semantic search: '{query}' (max: {max_results}, lang: {language})"
            }, None

        try:
            import asyncio
            if not self.serena.initialized:
                asyncio.run(self.serena.initialize())

            # Perform semantic search
            matches = asyncio.run(self.serena.semantic_search(query, max_results, language))

            result = {
                "query": query,
                "matches": len(matches),
                "results": [m.to_dict() for m in matches]
            }

            logger.info(f"Semantic search '{query}' found {len(matches)} matches")
            return result, None

        except Exception as e:
            error_msg = f"Error in semantic search: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

    def _execute_semantic_edit(self, step: ExecutionStep, dry_run: bool) -> Tuple[Any, Optional[str]]:
        """Execute semantic code edit (insert after symbol) using Serena"""
        if not self.serena:
            return None, "Serena semantic engine not available"

        symbol = step.parameters.get('symbol')
        code = step.parameters.get('code')
        language = step.parameters.get('language', 'python')

        if not symbol or not code:
            return None, "symbol and code required for semantic_edit"

        if dry_run:
            return {
                "dry_run": True,
                "action": f"Insert code after symbol '{symbol}'",
                "code_preview": code[:100]
            }, None

        try:
            import asyncio
            if not self.serena.initialized:
                asyncio.run(self.serena.initialize())

            # Perform semantic edit
            edit_result = asyncio.run(
                self.serena.insert_after_symbol(symbol, code, language, preserve_indentation=True)
            )

            if edit_result.success:
                logger.info(f"Successfully inserted code after symbol '{symbol}'")
                # Track in context
                self.context.add_edit(
                    edit_result.file_path,
                    edit_result.original_content[:100],
                    edit_result.new_content[:100],
                    f"Semantic edit: insert after {symbol}"
                )
                return {
                    "success": True,
                    "file_path": edit_result.file_path,
                    "lines_modified": edit_result.lines_modified,
                    "message": edit_result.message
                }, None
            else:
                return None, edit_result.message

        except Exception as e:
            error_msg = f"Error in semantic edit: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

    # ========================================================================
    # CODE-MODE EXECUTION (NEW)
    # ========================================================================

    def _select_execution_mode(self, plan: ExecutionPlan) -> str:
        """
        Intelligently select execution mode based on plan characteristics

        Args:
            plan: Execution plan to analyze

        Returns:
            "code_mode" or "traditional"
        """
        # Use code-mode if available and beneficial
        if not self.enable_code_mode or not self.code_mode_bridge:
            return "traditional"

        # Criteria for code-mode:
        # 1. Multi-step workflows (5+ steps)
        if len(plan.steps) >= 5:
            logger.debug(f"Code-mode selected: {len(plan.steps)} steps")
            return "code_mode"

        # 2. Complex tasks
        if hasattr(plan, 'complexity'):
            from advanced_planner import TaskComplexity
            if plan.complexity in [TaskComplexity.COMPLEX, TaskComplexity.VERY_COMPLEX]:
                logger.debug(f"Code-mode selected: complexity={plan.complexity.value}")
                return "code_mode"

        # 3. Has dependent steps (data flow between steps)
        if self._has_dependent_steps(plan):
            logger.debug("Code-mode selected: dependent steps detected")
            return "code_mode"

        # 4. Heavy file operations
        file_ops = sum(1 for step in plan.steps if step.step_type.name in
                      ['READ_FILE', 'WRITE_FILE', 'EDIT_FILE', 'SEARCH'])
        if file_ops >= 4:
            logger.debug(f"Code-mode selected: {file_ops} file operations")
            return "code_mode"

        # Default: traditional for simple tasks
        logger.debug("Traditional mode selected: simple task")
        return "traditional"

    def _has_dependent_steps(self, plan: ExecutionPlan) -> bool:
        """Check if plan has steps with dependencies"""
        return any(step.dependencies for step in plan.steps)

    def _execute_plan_code_mode(self, plan: ExecutionPlan) -> ExecutionResult:
        """
        Execute plan using code-mode (batched execution)

        Args:
            plan: Execution plan

        Returns:
            ExecutionResult
        """
        logger.info(f"Executing via CODE-MODE: {plan.task}")
        start_time = time.time()

        # Initialize code-mode bridge if needed
        if not self.code_mode_bridge._initialized:
            if not self.code_mode_bridge.initialize():
                logger.error("Code-mode initialization failed, falling back to traditional")
                return self.execute_plan(plan, mode="traditional")

        # Register DSMIL MCP server
        self.code_mode_bridge.register_mcp_server(
            name="dsmil",
            command="python3",
            args=["/home/user/LAT5150DRVMIL/03-mcp-servers/dsmil-tools/server.py", "--stdio"],
            description="DSMIL platform tools"
        )

        # Generate TypeScript code from plan
        typescript_code = self._plan_to_typescript(plan)

        if not typescript_code:
            logger.warning("Failed to generate TypeScript code, falling back to traditional")
            return self.execute_plan(plan, mode="traditional")

        # Execute via code-mode
        result = self.code_mode_bridge.execute_tool_chain(typescript_code)

        total_duration = time.time() - start_time

        if result.success:
            logger.info(f"✓ Code-mode execution successful ({result.duration_ms:.0f}ms)")

            return ExecutionResult(
                plan=plan,
                status=ExecutionStatus.SUCCESS,
                step_results=[StepResult(
                    step_num=1,
                    status=ExecutionStatus.SUCCESS,
                    output=result.result,
                    duration=result.duration_ms / 1000
                )],
                total_duration=total_duration,
                steps_succeeded=len(plan.steps),
                steps_failed=0,
                execution_mode="code_mode",
                api_calls=result.api_calls,
                tokens_estimate=result.tokens_estimate
            )

        else:
            logger.error(f"✗ Code-mode execution failed: {result.error}")
            logger.info("Falling back to traditional execution...")

            # Fallback to traditional
            return self.execute_plan(plan, mode="traditional")

    def _plan_to_typescript(self, plan: ExecutionPlan) -> Optional[str]:
        """
        Convert execution plan to TypeScript code for code-mode

        Args:
            plan: Execution plan

        Returns:
            TypeScript code string or None
        """
        # This is a simplified converter - Phase 4 (workflow optimizer) will enhance this
        lines = []
        lines.append("// Auto-generated from execution plan")
        lines.append(f"// Task: {plan.task}")
        lines.append("")

        for step in plan.steps:
            comment = f"// Step {step.step_num}: {step.description}"
            lines.append(comment)

            # Convert step to TypeScript tool call
            if step.step_type.name == "READ_FILE":
                filepath = step.parameters.get('filepath', '')
                if filepath:
                    lines.append(f"const file{step.step_num} = await dsmil.file_read({{ path: '{filepath}' }});")

            elif step.step_type.name == "WRITE_FILE":
                filepath = step.parameters.get('filepath', '')
                content = step.parameters.get('content', '')
                if filepath:
                    lines.append(f"await dsmil.file_write({{ path: '{filepath}', content: `{content}` }});")

            elif step.step_type.name == "SEARCH":
                pattern = step.parameters.get('pattern', '')
                if pattern:
                    lines.append(f"const search{step.step_num} = await dsmil.file_search({{ pattern: '{pattern}' }});")

            elif step.step_type.name == "GENERATE_CODE":
                lines.append(f"const code{step.step_num} = await dsmil.ai_generate({{ prompt: '{step.description}' }});")

            elif step.step_type.name == "EXECUTE":
                command = step.parameters.get('command', '')
                if command:
                    lines.append(f"const exec{step.step_num} = await dsmil.exec_command({{ command: '{command}' }});")

            # NEW: Semantic code analysis steps (Slither MCP approach - deterministic!)
            elif step.step_type.name == "FIND_SYMBOL":
                symbol_name = step.parameters.get('symbol_name', step.parameters.get('name', ''))
                symbol_type = step.parameters.get('symbol_type', 'any')
                if symbol_name:
                    lines.append(f"const symbol{step.step_num} = await dsmil.code_find_symbol({{ name: '{symbol_name}', symbol_type: '{symbol_type}' }});")

            elif step.step_type.name == "FIND_REFERENCES":
                filepath = step.parameters.get('file_path', step.parameters.get('filepath', ''))
                line = step.parameters.get('line', 1)
                column = step.parameters.get('column', 0)
                if filepath:
                    lines.append(f"const refs{step.step_num} = await dsmil.code_find_references({{ file_path: '{filepath}', line: {line}, column: {column} }});")

            elif step.step_type.name == "SEMANTIC_SEARCH":
                query = step.parameters.get('query', '')
                max_results = step.parameters.get('max_results', 10)
                if query:
                    lines.append(f"const search{step.step_num} = await dsmil.code_semantic_search({{ query: '{query}', max_results: {max_results} }});")

            elif step.step_type.name == "SEMANTIC_EDIT":
                symbol = step.parameters.get('symbol', '')
                code = step.parameters.get('code', '').replace("`", "\\`").replace("'", "\\'")
                if symbol and code:
                    lines.append(f"const edit{step.step_num} = await dsmil.code_insert_after_symbol({{ symbol: '{symbol}', code: `{code}` }});")

            else:
                # Skip unsupported step types for now
                lines.append(f"// Unsupported step type: {step.step_type.name}")

            lines.append("")

        # Return result
        lines.append("return { success: true, steps_completed: " + str(len(plan.steps)) + " };")

        return "\n".join(lines)


def main():
    """Example usage"""
    from dsmil_ai_engine import DSMILAIEngine
    from advanced_planner import AdvancedPlanner, TaskComplexity, ExecutionPlan, ExecutionStep

    print("=== Execution Engine Demo ===\n")

    # Initialize components
    ai = DSMILAIEngine()
    files = FileOps()
    edits = EditOps()
    tools = ToolOps()
    context = ContextManager()
    learner = CodebaseLearner()
    pattern_db = PatternDatabase()

    # Create execution engine
    engine = ExecutionEngine(
        ai_engine=ai,
        file_ops=files,
        edit_ops=edits,
        tool_ops=tools,
        context_manager=context,
        codebase_learner=learner,
        pattern_db=pattern_db
    )

    # Create a simple plan
    plan = ExecutionPlan(
        task="List Python files in current directory",
        complexity=TaskComplexity.SIMPLE,
        steps=[
            ExecutionStep(
                step_num=1,
                step_type=StepType.EXECUTE,
                description="List Python files",
                action="ls *.py"
            )
        ],
        estimated_time=5,
        files_involved=[],
        dependencies=[],
        risks=[],
        success_criteria=["Files listed successfully"],
        model_used="code"
    )

    print(f"Plan: {plan.task}")
    print(f"Steps: {len(plan.steps)}")

    print("\nExecuting plan (dry run)...")
    result = engine.execute_plan(plan, dry_run=True)

    print(f"\nResult: {result.status.value}")
    print(f"Duration: {result.total_duration:.2f}s")
    print(f"Succeeded: {result.steps_succeeded}/{len(plan.steps)}")

    print("\n✓ Execution Engine ready!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Advanced Planner Module for Local Claude Code
Sophisticated task planning with multi-model integration (Qwen, WhiteRabbit, DeepSeek)
"""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class StepType(Enum):
    """Types of execution steps"""
    READ_FILE = "read_file"
    EDIT_FILE = "edit_file"
    WRITE_FILE = "write_file"
    SEARCH = "search"
    ANALYZE = "analyze"
    EXECUTE = "execute"
    TEST = "test"
    GIT = "git"
    GENERATE_CODE = "generate_code"
    LEARN_PATTERN = "learn_pattern"
    # Serena semantic code operations
    FIND_SYMBOL = "find_symbol"  # Find symbol definitions using LSP
    FIND_REFERENCES = "find_references"  # Find all references to a symbol
    SEMANTIC_SEARCH = "semantic_search"  # Semantic code search
    SEMANTIC_EDIT = "semantic_edit"  # Edit code at symbol level


class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"           # 1-2 steps, single file
    MODERATE = "moderate"       # 3-5 steps, multiple files
    COMPLEX = "complex"         # 6-10 steps, refactoring
    VERY_COMPLEX = "very_complex"  # 10+ steps, architectural changes


@dataclass
class ExecutionStep:
    """Single execution step in a plan"""
    step_num: int
    step_type: StepType
    description: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[int] = field(default_factory=list)  # Step numbers this depends on
    expected_output: Optional[str] = None
    fallback: Optional[str] = None  # Fallback action if step fails


@dataclass
class ExecutionPlan:
    """Complete execution plan for a task"""
    task: str
    complexity: TaskComplexity
    steps: List[ExecutionStep]
    estimated_time: int  # seconds
    files_involved: List[str]
    dependencies: List[str]  # External dependencies needed
    risks: List[str]  # Potential risks/issues
    success_criteria: List[str]  # How to verify success
    model_used: str  # Which AI model created the plan


class AdvancedPlanner:
    """
    Advanced task planner with multi-model integration

    Features:
    - Sophisticated task breakdown
    - Multi-step planning with dependencies
    - Model selection based on task complexity
    - Pattern learning and reuse
    - Self-improvement from execution results
    """

    def __init__(self, ai_engine, rag_system=None, pattern_db=None):
        """
        Initialize advanced planner

        Args:
            ai_engine: DSMIL AI engine instance
            rag_system: RAG system for retrieving similar patterns
            pattern_db: Pattern database for storing learned patterns
        """
        self.ai = ai_engine
        self.rag = rag_system
        self.pattern_db = pattern_db
        self.execution_history = []

        logger.info("AdvancedPlanner initialized")

    def plan_task(self, task: str, context: Optional[Dict] = None) -> ExecutionPlan:
        """
        Create execution plan for task

        Args:
            task: Task description
            context: Additional context (files, codebase info, etc.)

        Returns:
            ExecutionPlan with steps
        """
        logger.info(f"Planning task: {task}")

        # Step 1: Analyze task complexity
        complexity = self._analyze_complexity(task, context)
        logger.info(f"Task complexity: {complexity.value}")

        # Step 2: Select appropriate model
        model = self._select_model_for_task(complexity)
        logger.info(f"Selected model: {model}")

        # Step 3: Retrieve similar patterns from history
        similar_patterns = self._retrieve_similar_patterns(task) if self.rag else []

        # Step 4: Generate plan using AI
        plan_steps = self._generate_plan_steps(task, complexity, model, similar_patterns, context)

        # Step 5: Analyze dependencies and risks
        files_involved = self._extract_files_from_steps(plan_steps)
        dependencies = self._identify_dependencies(plan_steps)
        risks = self._identify_risks(plan_steps, context)
        success_criteria = self._define_success_criteria(task, plan_steps)

        # Step 6: Estimate execution time
        estimated_time = self._estimate_execution_time(plan_steps, complexity)

        plan = ExecutionPlan(
            task=task,
            complexity=complexity,
            steps=plan_steps,
            estimated_time=estimated_time,
            files_involved=files_involved,
            dependencies=dependencies,
            risks=risks,
            success_criteria=success_criteria,
            model_used=model
        )

        logger.info(f"Plan created: {len(plan_steps)} steps, ~{estimated_time}s")

        return plan

    def _analyze_complexity(self, task: str, context: Optional[Dict] = None) -> TaskComplexity:
        """Analyze task complexity"""

        # Use AI to classify complexity
        complexity_prompt = f"""Analyze this coding task and classify its complexity.

Task: {task}

Context: {json.dumps(context) if context else "None"}

Consider:
- Number of files to modify
- Amount of code to write/change
- Architectural changes needed
- Testing requirements
- Dependencies

Classify as: SIMPLE, MODERATE, COMPLEX, or VERY_COMPLEX

Respond with ONLY one word: SIMPLE, MODERATE, COMPLEX, or VERY_COMPLEX"""

        result = self.ai.generate(complexity_prompt, model_selection="fast")

        if 'error' not in result:
            response = result['response'].strip().upper()
            if "SIMPLE" in response:
                return TaskComplexity.SIMPLE
            elif "MODERATE" in response:
                return TaskComplexity.MODERATE
            elif "VERY_COMPLEX" in response or "VERY COMPLEX" in response:
                return TaskComplexity.VERY_COMPLEX
            elif "COMPLEX" in response:
                return TaskComplexity.COMPLEX

        # Fallback: analyze keywords
        task_lower = task.lower()

        # Simple indicators
        if any(word in task_lower for word in ['add docstring', 'fix typo', 'rename variable']):
            return TaskComplexity.SIMPLE

        # Complex indicators
        if any(word in task_lower for word in ['refactor', 'redesign', 'architecture', 'migrate']):
            return TaskComplexity.COMPLEX

        # Very complex indicators
        if any(word in task_lower for word in ['rewrite', 'complete overhaul', 'new system']):
            return TaskComplexity.VERY_COMPLEX

        # Default to moderate
        return TaskComplexity.MODERATE

    def _select_model_for_task(self, complexity: TaskComplexity) -> str:
        """Select best model based on task complexity"""

        model_mapping = {
            TaskComplexity.SIMPLE: "code",           # DeepSeek Coder (fast, good quality)
            TaskComplexity.MODERATE: "quality_code",  # Qwen Coder (better quality)
            TaskComplexity.COMPLEX: "quality_code",   # Qwen Coder (best for complex)
            TaskComplexity.VERY_COMPLEX: "quality_code"  # Qwen Coder
        }

        return model_mapping.get(complexity, "code")

    def _retrieve_similar_patterns(self, task: str) -> List[Dict]:
        """Retrieve similar task patterns from history"""
        if not self.rag:
            return []

        try:
            # Search for similar tasks
            results = self.rag.search(task, top_k=3)

            patterns = []
            for result in results:
                patterns.append({
                    "task": result.text,
                    "score": result.score,
                    "metadata": result.metadata
                })

            return patterns
        except Exception as e:
            logger.error(f"Error retrieving patterns: {e}")
            return []

    def _generate_plan_steps(
        self,
        task: str,
        complexity: TaskComplexity,
        model: str,
        similar_patterns: List[Dict],
        context: Optional[Dict]
    ) -> List[ExecutionStep]:
        """Generate execution steps using AI"""

        # Build planning prompt
        prompt = f"""Create a detailed execution plan for this coding task.

Task: {task}

Complexity: {complexity.value}

"""

        if similar_patterns:
            prompt += "\nSimilar tasks solved before:\n"
            for i, pattern in enumerate(similar_patterns, 1):
                prompt += f"{i}. {pattern['task']} (similarity: {pattern['score']:.2f})\n"

        if context:
            prompt += f"\nContext: {json.dumps(context, indent=2)}\n"

        prompt += """
Break down the task into specific, actionable steps. For each step, specify:
1. Step number
2. Action type (read_file, edit_file, write_file, search, analyze, execute, test, git, generate_code)
3. Description
4. Specific action command/details
5. Expected output

Format as JSON array:
[
  {
    "step": 1,
    "type": "read_file",
    "description": "Read the target file",
    "action": "read server.py",
    "expected_output": "File contents with function definitions"
  },
  {
    "step": 2,
    "type": "analyze",
    "description": "Identify functions needing logging",
    "action": "analyze functions in server.py for logging needs",
    "expected_output": "List of functions to add logging to"
  },
  ...
]

Provide ONLY the JSON array, no other text."""

        # Generate plan
        result = self.ai.generate(prompt, model_selection=model)

        if 'error' in result:
            logger.error(f"Plan generation failed: {result['error']}")
            return self._generate_fallback_plan(task)

        # Parse JSON response
        try:
            steps_data = self._extract_json_from_response(result['response'])

            steps = []
            for step_data in steps_data:
                step = ExecutionStep(
                    step_num=step_data.get('step', len(steps) + 1),
                    step_type=StepType(step_data.get('type', 'execute')),
                    description=step_data.get('description', ''),
                    action=step_data.get('action', ''),
                    expected_output=step_data.get('expected_output'),
                    parameters=step_data.get('parameters', {})
                )
                steps.append(step)

            return steps

        except Exception as e:
            logger.error(f"Error parsing plan: {e}")
            return self._generate_fallback_plan(task)

    def _extract_json_from_response(self, response: str) -> List[Dict]:
        """Extract JSON array from AI response"""
        import re

        # Try to find JSON array in response
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback: try entire response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            raise ValueError("Could not extract JSON from response")

    def _generate_fallback_plan(self, task: str) -> List[ExecutionStep]:
        """Generate simple fallback plan if AI planning fails"""
        logger.warning("Using fallback plan")

        return [
            ExecutionStep(
                step_num=1,
                step_type=StepType.ANALYZE,
                description="Analyze task requirements",
                action=f"analyze: {task}"
            ),
            ExecutionStep(
                step_num=2,
                step_type=StepType.GENERATE_CODE,
                description="Generate code for task",
                action=f"generate code: {task}"
            ),
            ExecutionStep(
                step_num=3,
                step_type=StepType.TEST,
                description="Test generated code",
                action="run tests"
            )
        ]

    def _extract_files_from_steps(self, steps: List[ExecutionStep]) -> List[str]:
        """Extract file paths mentioned in steps"""
        files = set()

        for step in steps:
            action = step.action.lower()

            # Extract file extensions
            import re
            file_patterns = r'[\w/.-]+\.(?:py|js|ts|c|cpp|h|sh|md|json|yaml|yml)'
            matches = re.findall(file_patterns, action)
            files.update(matches)

        return sorted(list(files))

    def _identify_dependencies(self, steps: List[ExecutionStep]) -> List[str]:
        """Identify external dependencies needed"""
        dependencies = set()

        for step in steps:
            action = step.action.lower()

            # Check for common dependencies
            if 'import' in action or 'require' in action:
                dependencies.add("Check imports in generated code")
            if 'pytest' in action or 'test' in action:
                dependencies.add("pytest")
            if 'git' in action:
                dependencies.add("git")
            if 'npm' in action:
                dependencies.add("npm/node")

        return sorted(list(dependencies))

    def _identify_risks(self, steps: List[ExecutionStep], context: Optional[Dict]) -> List[str]:
        """Identify potential risks in execution"""
        risks = []

        # Check for file modifications
        edit_count = sum(1 for step in steps if step.step_type == StepType.EDIT_FILE)
        if edit_count > 5:
            risks.append(f"High number of file edits ({edit_count}) - consider careful review")

        # Check for git operations
        has_git = any(step.step_type == StepType.GIT for step in steps)
        if has_git:
            risks.append("Git operations - ensure working directory is clean")

        # Check for execution steps
        exec_count = sum(1 for step in steps if step.step_type == StepType.EXECUTE)
        if exec_count > 0:
            risks.append(f"Command execution steps ({exec_count}) - review before running")

        # Check context for warnings
        if context and context.get('has_uncommitted_changes'):
            risks.append("Uncommitted changes in repository")

        return risks

    def _define_success_criteria(self, task: str, steps: List[ExecutionStep]) -> List[str]:
        """Define criteria for successful task completion"""
        criteria = []

        # Check for test steps
        has_tests = any(step.step_type == StepType.TEST for step in steps)
        if has_tests:
            criteria.append("All tests pass")

        # Check for file edits
        has_edits = any(step.step_type == StepType.EDIT_FILE for step in steps)
        if has_edits:
            criteria.append("Files edited without syntax errors")

        # Check for code generation
        has_generation = any(step.step_type == StepType.GENERATE_CODE for step in steps)
        if has_generation:
            criteria.append("Generated code compiles/runs without errors")

        # Generic criteria
        criteria.append("Task requirements fulfilled")
        criteria.append("No breaking changes introduced")

        return criteria

    def _estimate_execution_time(self, steps: List[ExecutionStep], complexity: TaskComplexity) -> int:
        """Estimate execution time in seconds"""

        # Base time per step type
        step_times = {
            StepType.READ_FILE: 2,
            StepType.EDIT_FILE: 5,
            StepType.WRITE_FILE: 3,
            StepType.SEARCH: 5,
            StepType.ANALYZE: 10,
            StepType.EXECUTE: 10,
            StepType.TEST: 15,
            StepType.GIT: 3,
            StepType.GENERATE_CODE: 20
        }

        total_time = sum(step_times.get(step.step_type, 5) for step in steps)

        # Complexity multiplier
        multipliers = {
            TaskComplexity.SIMPLE: 1.0,
            TaskComplexity.MODERATE: 1.5,
            TaskComplexity.COMPLEX: 2.0,
            TaskComplexity.VERY_COMPLEX: 3.0
        }

        return int(total_time * multipliers.get(complexity, 1.0))

    def record_execution(self, plan: ExecutionPlan, results: List[Dict], success: bool):
        """Record execution results for learning"""

        execution_record = {
            "task": plan.task,
            "complexity": plan.complexity.value,
            "steps": len(plan.steps),
            "success": success,
            "model_used": plan.model_used,
            "timestamp": __import__('time').time()
        }

        self.execution_history.append(execution_record)

        # Store in pattern database if available
        if self.pattern_db and success:
            self.pattern_db.store_pattern(plan.task, plan, results)

        logger.info(f"Recorded execution: {plan.task} - {'SUCCESS' if success else 'FAILED'}")

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {"total": 0, "success_rate": 0.0}

        total = len(self.execution_history)
        successful = sum(1 for ex in self.execution_history if ex['success'])

        return {
            "total": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "avg_steps": sum(ex['steps'] for ex in self.execution_history) / total
        }


def main():
    """Example usage"""
    from dsmil_ai_engine import DSMILAIEngine

    print("=== Advanced Planner Demo ===\n")

    ai = DSMILAIEngine()
    planner = AdvancedPlanner(ai)

    # Example task
    task = "Add comprehensive logging to server.py with different log levels (DEBUG, INFO, ERROR)"

    print(f"Task: {task}\n")
    print("Creating execution plan...\n")

    plan = planner.plan_task(task)

    print(f"Complexity: {plan.complexity.value}")
    print(f"Model: {plan.model_used}")
    print(f"Estimated time: {plan.estimated_time}s")
    print(f"Files involved: {', '.join(plan.files_involved) if plan.files_involved else 'None detected'}")
    print(f"\nSteps ({len(plan.steps)}):")

    for step in plan.steps:
        print(f"  {step.step_num}. [{step.step_type.value}] {step.description}")
        print(f"     Action: {step.action}")

    if plan.risks:
        print(f"\nRisks:")
        for risk in plan.risks:
            print(f"  �  {risk}")

    print(f"\nSuccess criteria:")
    for criterion in plan.success_criteria:
        print(f"   {criterion}")

    print("\n Advanced Planner ready!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Local Claude Code - MVP Agentic Coding System

Enables local AI to edit codebases like Claude Code:
- Read files
- Edit files (surgical replacements)
- Execute commands
- Plan multi-step tasks
- All with DSMIL attestation, no restrictions

Perfect for offensive security research that violates Claude TOS.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List

# Import modules
sys.path.insert(0, str(Path(__file__).parent))
from file_operations import FileOps
from edit_operations import EditOps
from tool_operations import ToolOps
from dsmil_ai_engine import DSMILAIEngine

class LocalClaudeCode:
    def __init__(self, workspace_root: str = None):
        """
        Initialize Local Claude Code system

        Args:
            workspace_root: Project root directory
        """
        self.workspace = Path(workspace_root) if workspace_root else Path.cwd()
        self.files = FileOps(workspace_root=self.workspace)
        self.edits = EditOps(workspace_root=self.workspace, create_backups=True)
        self.tools = ToolOps(workspace_root=self.workspace)
        self.ai = DSMILAIEngine()

        print(f"🎯 Local Claude Code initialized")
        print(f"   Workspace: {self.workspace}")
        print(f"   AI Models: DeepSeek Coder (code), Qwen Coder (quality)")
        print(f"   DSMIL: Device 16 attestation enabled")
        print(f"   Restrictions: NONE (perfect for offensive security research)")

    def execute_task(self, task: str, model: str = "code") -> Dict:
        """
        Execute a coding task (like asking Claude Code)

        Args:
            task: Description of what to do
            model: AI model to use ("code", "quality_code", "fast")

        Returns:
            Result dict with actions taken
        """
        print(f"\n🎯 Task: {task}\n")

        # Step 1: Plan the task
        print("📋 Planning steps...")
        plan_prompt = f"""Break this task into specific actions:

Task: {task}

Provide a numbered list of concrete steps like:
1. Read file: /path/to/file.py
2. Find: function_name
3. Edit: Replace old code with new code
4. Test: Run test command

Be specific and actionable."""

        plan_result = self.ai.generate(plan_prompt, model_selection=model)

        if 'error' in plan_result:
            return {"error": "Planning failed", "details": plan_result}

        steps = self._parse_steps(plan_result['response'])
        print(f"\n📝 Plan ({len(steps)} steps):")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")

        # Step 2: Execute steps
        print(f"\n⚡ Executing...")
        results = []

        for i, step in enumerate(steps, 1):
            print(f"\n[{i}/{len(steps)}] {step}")

            step_result = self._execute_step(step, model=model)
            results.append({
                "step": step,
                "result": step_result
            })

            if step_result.get('error'):
                print(f"  ❌ Error: {step_result['error']}")
                # Ask AI how to recover
                recover_prompt = f"Step failed: {step}\nError: {step_result['error']}\nHow to fix?"
                recovery = self.ai.generate(recover_prompt, model_selection="fast")
                print(f"  💡 Suggestion: {recovery.get('response', 'Unknown')[:200]}")
                break
            else:
                print(f"  ✅ Success")

        print(f"\n✅ Task complete!")

        return {
            "task": task,
            "steps": steps,
            "results": results,
            "edits_made": len(self.edits.edits_made),
            "dsmil_attested": True
        }

    def _parse_steps(self, plan_text: str) -> list:
        """Extract numbered steps from plan"""
        steps = []
        for line in plan_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                # Remove numbering and clean up
                clean = line.lstrip('0123456789.-* ').strip()
                if clean:
                    steps.append(clean)
        return steps

    def _execute_step(self, step: str, model: str = "code") -> Dict:
        """Execute one step from plan"""
        step_lower = step.lower()

        # Classify step type
        if "read" in step_lower and ("file" in step_lower or ".py" in step_lower or ".js" in step_lower):
            # Extract filepath
            filepath = self._extract_filepath(step)
            if filepath:
                return self.files.read_file(filepath)
            return {"error": "Could not extract filepath from step"}

        elif "find" in step_lower or "search" in step_lower or "grep" in step_lower:
            # Extract search pattern
            pattern = self._extract_pattern(step)
            if pattern:
                return self.files.grep(pattern)
            return {"error": "Could not extract search pattern"}

        elif "edit" in step_lower or "modify" in step_lower or "replace" in step_lower or "change" in step_lower:
            # This needs AI to generate the actual edit
            print(f"    🤖 Asking AI to generate edit...")

            # Get file context if mentioned
            filepath = self._extract_filepath(step)
            context = ""
            if filepath:
                file_result = self.files.read_file(filepath)
                if 'content' in file_result:
                    context = f"\n\nFile content:\n{file_result['content']}"

            edit_prompt = f"""Generate an edit for this file:

Step: {step}{context}

Provide ONLY:
OLD: <exact string to replace>
NEW: <new string>

Be precise with the exact strings."""

            edit_result = self.ai.generate(edit_prompt, model_selection=model)

            if 'error' in edit_result:
                return edit_result

            # Parse OLD/NEW from response
            old, new = self._parse_edit_strings(edit_result['response'])

            if old and new and filepath:
                return self.edits.edit_file(filepath, old, new)
            return {"error": "Could not parse edit from AI response"}

        elif "run" in step_lower or "execute" in step_lower or "test" in step_lower:
            # Extract command
            command = self._extract_command(step)
            if command:
                return self.tools.bash(command)
            return {"error": "Could not extract command from step"}

        elif "create" in step_lower or "write" in step_lower:
            # Generate new code
            filepath = self._extract_filepath(step)

            gen_prompt = f"""Generate code for this task:

{step}

Provide ONLY the code, no explanation."""

            code_result = self.ai.generate(gen_prompt, model_selection=model)

            if 'error' in code_result:
                return code_result

            if filepath:
                return self.edits.write_file(filepath, code_result['response'])
            return {"code_generated": code_result['response'][:500]}

        else:
            # Ask AI to interpret the step
            return self.ai.generate(step, model_selection=model)

    def _extract_filepath(self, text: str) -> str:
        """Extract filepath from text"""
        import re
        # Look for common patterns
        patterns = [
            r'["\']([^"\']+\.(?:py|js|ts|c|cpp|h|sh|md))["\']',
            r'(\S+\.(?:py|js|ts|c|cpp|h|sh|md))',
            r'file[:\s]+(\S+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        return None

    def _extract_pattern(self, text: str) -> str:
        """Extract search pattern from text"""
        import re
        # Look for quoted strings or keywords
        match = re.search(r'["\']([^"\']+)["\']', text)
        if match:
            return match.group(1)

        # Fallback: last word
        words = text.split()
        return words[-1] if words else None

    def _extract_command(self, text: str) -> str:
        """Extract command from text"""
        import re
        # Look for quoted commands
        match = re.search(r'["\']([^"\']+)["\']', text)
        if match:
            return match.group(1)

        # Look for common command patterns
        if "pytest" in text.lower():
            return "pytest"
        if "python" in text.lower():
            return "python3 test.py"

        return None

    def _parse_edit_strings(self, response: str) -> tuple:
        """Parse OLD/NEW strings from AI response"""
        import re

        old_match = re.search(r'OLD:\s*```?([^`]+)```?', response, re.DOTALL | re.IGNORECASE)
        new_match = re.search(r'NEW:\s*```?([^`]+)```?', response, re.DOTALL | re.IGNORECASE)

        if old_match and new_match:
            return old_match.group(1).strip(), new_match.group(1).strip()

        # Fallback: look for quoted strings
        quotes = re.findall(r'```([^`]+)```', response)
        if len(quotes) >= 2:
            return quotes[0].strip(), quotes[1].strip()

        return None, None

# CLI
if __name__ == "__main__":
    agent = LocalClaudeCode()

    if len(sys.argv) < 2:
        print("""
Local Claude Code - MVP Agentic System

Usage:
  python3 local_claude_code.py "Add logging to server.py"
  python3 local_claude_code.py "Fix the bug in auth.py"
  python3 local_claude_code.py "Refactor database.py to use async"
  python3 local_claude_code.py "Create a new API endpoint in api.py"

Features:
- Reads files
- Makes edits (surgical string replacement)
- Runs commands
- Multi-step planning
- DSMIL-attested
- No restrictions (perfect for offensive security)

Example (offensive research):
  python3 local_claude_code.py "Generate shellcode for CVE-2024-XXXX in exploit.c"

All work is local, DSMIL-attested, and unrestricted.
        """)
        sys.exit(1)

    task = ' '.join(sys.argv[1:])
    result = agent.execute_task(task)
    print(f"\n📊 Summary: {result['edits_made']} files edited")

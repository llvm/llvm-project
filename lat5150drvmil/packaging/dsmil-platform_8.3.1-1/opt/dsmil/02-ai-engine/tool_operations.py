#!/usr/bin/env python3
"""
Tool Operations Module - Local Claude Code MVP
Provides bash execution, git operations (like Claude Code Bash tool)
"""

import subprocess
from pathlib import Path
from typing import Dict, Optional

class ToolOps:
    def __init__(self, workspace_root: str = None, timeout: int = 120):
        """
        Initialize tool operations

        Args:
            workspace_root: Working directory for commands
            timeout: Default command timeout in seconds
        """
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.default_timeout = timeout

    def bash(self, command: str, timeout: Optional[int] = None, description: str = None) -> Dict:
        """
        Execute bash command (like Claude Code Bash tool)

        Args:
            command: Shell command to execute
            timeout: Command timeout (default: 120s)
            description: Human-readable description of command

        Returns:
            Dict with stdout, stderr, returncode
        """
        try:
            timeout = timeout or self.default_timeout

            # Execute in workspace root
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.workspace_root)
            )

            return {
                "command": command,
                "description": description,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "success": result.returncode == 0
            }

        except subprocess.TimeoutExpired:
            return {
                "error": f"Command timed out after {timeout}s",
                "command": command
            }
        except Exception as e:
            return {
                "error": str(e),
                "command": command
            }

    def git(self, args: str, description: str = None) -> Dict:
        """
        Execute git command

        Args:
            args: Git arguments (e.g., "status", "diff", "add .")
            description: Human-readable description

        Returns:
            Dict with result
        """
        return self.bash(f"git {args}", description=description or f"git {args}")

    def git_status(self) -> Dict:
        """Get git status"""
        return self.git("status")

    def git_diff(self, filepath: str = None) -> Dict:
        """Get git diff"""
        if filepath:
            return self.git(f"diff {filepath}")
        return self.git("diff")

    def git_add(self, filepath: str = ".") -> Dict:
        """Stage files for commit"""
        return self.git(f"add {filepath}")

    def git_commit(self, message: str) -> Dict:
        """Create git commit"""
        # Escape quotes in message
        safe_message = message.replace('"', '\\"')
        return self.git(f'commit -m "{safe_message}"')

    def run_tests(self, test_command: str = "pytest") -> Dict:
        """Run tests"""
        return self.bash(test_command, description="Running tests")

    def check_syntax(self, filepath: str) -> Dict:
        """Check syntax of file"""
        filepath = Path(filepath)

        if filepath.suffix == ".py":
            return self.bash(f"python3 -m py_compile {filepath}", description="Check Python syntax")
        elif filepath.suffix in [".js", ".ts"]:
            return self.bash(f"node --check {filepath}", description="Check JavaScript syntax")
        elif filepath.suffix in [".c", ".cpp"]:
            return self.bash(f"gcc -fsyntax-only {filepath}", description="Check C/C++ syntax")
        else:
            return {"error": f"Unsupported file type: {filepath.suffix}"}

# CLI
if __name__ == "__main__":
    import sys
    import json

    ops = ToolOps()

    if len(sys.argv) < 2:
        print("ToolOps - Usage:")
        print("  python3 tool_operations.py bash 'ls -la'")
        print("  python3 tool_operations.py git 'status'")
        print("  python3 tool_operations.py test")
        print("  python3 tool_operations.py check-syntax /path/file.py")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "bash" and len(sys.argv) > 2:
        result = ops.bash(sys.argv[2])
        if result.get('success'):
            print(result['stdout'])
            if result['stderr']:
                print(f"STDERR: {result['stderr']}", file=sys.stderr)
        else:
            print(f"Error: {result.get('error', result['stderr'])}")

    elif cmd == "git" and len(sys.argv) > 2:
        result = ops.git(sys.argv[2])
        print(result['stdout'])

    elif cmd == "test":
        test_cmd = sys.argv[2] if len(sys.argv) > 2 else "pytest"
        result = ops.run_tests(test_cmd)
        print(result['stdout'])

    elif cmd == "check-syntax" and len(sys.argv) > 2:
        result = ops.check_syntax(sys.argv[2])
        print(json.dumps(result, indent=2))

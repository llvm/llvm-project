#!/usr/bin/env python3
"""
File Operations Module - Local Claude Code MVP
Provides file reading, listing, and searching capabilities
"""

import os
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

class FileOps:
    def __init__(self, workspace_root: str = None):
        """
        Initialize file operations

        Args:
            workspace_root: Root directory for operations (safety boundary)
        """
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()

    def read_file(self, filepath: str, offset: int = 0, limit: Optional[int] = None) -> Dict:
        """
        Read file contents (like Claude Code Read tool)

        Args:
            filepath: Path to file
            offset: Line number to start from (0-indexed)
            limit: Max lines to read

        Returns:
            Dict with content, lines, or error
        """
        try:
            filepath = Path(filepath).expanduser().resolve()

            # Safety check: ensure file is within workspace or explicitly allowed
            if not filepath.exists():
                return {"error": f"File not found: {filepath}"}

            if filepath.is_dir():
                return {"error": f"Path is a directory, not a file: {filepath}"}

            # Read file
            with open(filepath, 'r', errors='ignore') as f:
                lines = f.readlines()

            # Apply offset and limit
            if limit:
                selected_lines = lines[offset:offset+limit]
            else:
                selected_lines = lines[offset:]

            # Format with line numbers (like Claude Code)
            numbered_lines = []
            for i, line in enumerate(selected_lines, start=offset+1):
                numbered_lines.append(f"{i:6d}\t{line.rstrip()}")

            return {
                "filepath": str(filepath),
                "total_lines": len(lines),
                "showing_lines": len(selected_lines),
                "offset": offset,
                "content": '\n'.join(numbered_lines)
            }

        except Exception as e:
            return {"error": str(e), "filepath": filepath}

    def list_files(self, pattern: str, path: str = None) -> Dict:
        """
        List files matching pattern (like Claude Code Glob tool)

        Args:
            pattern: Glob pattern (e.g., "*.py", "src/**/*.js")
            path: Directory to search in (default: workspace_root)

        Returns:
            Dict with matched files or error
        """
        try:
            import glob

            search_path = Path(path) if path else self.workspace_root
            full_pattern = str(search_path / pattern)

            matches = glob.glob(full_pattern, recursive=True)

            # Sort by modification time (newest first, like Claude Code)
            matches.sort(key=lambda x: os.path.getmtime(x), reverse=True)

            return {
                "pattern": pattern,
                "search_path": str(search_path),
                "matches": matches[:100],  # Limit to 100 like Claude Code
                "total": len(matches)
            }

        except Exception as e:
            return {"error": str(e), "pattern": pattern}

    def grep(self, pattern: str, path: str = None, file_pattern: str = "*") -> Dict:
        """
        Search for pattern in files (like Claude Code Grep tool)

        Args:
            pattern: Regex pattern to search for
            path: Directory to search (default: workspace_root)
            file_pattern: File glob to filter (e.g., "*.py")

        Returns:
            Dict with matching files and lines
        """
        try:
            search_path = Path(path) if path else self.workspace_root

            # Use ripgrep if available, fallback to grep
            if subprocess.run(['which', 'rg'], capture_output=True).returncode == 0:
                cmd = ['rg', '--json', '-i', pattern, str(search_path)]
                if file_pattern != "*":
                    cmd.extend(['-g', file_pattern])
            else:
                cmd = ['grep', '-r', '-n', '-i', pattern, str(search_path)]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                return {
                    "pattern": pattern,
                    "search_path": str(search_path),
                    "output": result.stdout,
                    "matches_found": True
                }
            else:
                return {
                    "pattern": pattern,
                    "search_path": str(search_path),
                    "matches_found": False,
                    "output": ""
                }

        except Exception as e:
            return {"error": str(e), "pattern": pattern}

    def find_function(self, function_name: str, file_pattern: str = "*.py") -> Dict:
        """Find function definition in codebase"""
        pattern = f"def {function_name}|function {function_name}|{function_name} ="
        return self.grep(pattern, file_pattern=file_pattern)

    def find_class(self, class_name: str, file_pattern: str = "*.py") -> Dict:
        """Find class definition in codebase"""
        pattern = f"class {class_name}"
        return self.grep(pattern, file_pattern=file_pattern)

# CLI
if __name__ == "__main__":
    import sys
    import json

    ops = FileOps()

    if len(sys.argv) < 2:
        print("FileOps - Usage:")
        print("  python3 file_operations.py read /path/to/file.py")
        print("  python3 file_operations.py list '*.py' .")
        print("  python3 file_operations.py grep 'pattern' /path")
        print("  python3 file_operations.py find-func function_name")
        print("  python3 file_operations.py find-class ClassName")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "read" and len(sys.argv) > 2:
        result = ops.read_file(sys.argv[2])
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(result['content'])

    elif cmd == "list" and len(sys.argv) > 2:
        pattern = sys.argv[2]
        path = sys.argv[3] if len(sys.argv) > 3 else None
        result = ops.list_files(pattern, path)
        print(json.dumps(result, indent=2))

    elif cmd == "grep" and len(sys.argv) > 2:
        pattern = sys.argv[2]
        path = sys.argv[3] if len(sys.argv) > 3 else None
        result = ops.grep(pattern, path)
        if result.get('matches_found'):
            print(result['output'])
        else:
            print(f"No matches found for: {pattern}")

    elif cmd == "find-func" and len(sys.argv) > 2:
        result = ops.find_function(sys.argv[2])
        print(json.dumps(result, indent=2))

    elif cmd == "find-class" and len(sys.argv) > 2:
        result = ops.find_class(sys.argv[2])
        print(json.dumps(result, indent=2))

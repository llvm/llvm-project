#!/usr/bin/env python3
"""
Edit Operations Module - Local Claude Code MVP
Provides file editing and writing capabilities (like Claude Code Edit/Write tools)
"""

import os
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

class EditOps:
    def __init__(self, workspace_root: str = None, create_backups: bool = True):
        """
        Initialize edit operations

        Args:
            workspace_root: Root directory for operations
            create_backups: Create .bak files before edits
        """
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.create_backups = create_backups
        self.edits_made = []

    def edit_file(self, filepath: str, old_string: str, new_string: str, replace_all: bool = False) -> Dict:
        """
        Edit file by replacing string (like Claude Code Edit tool)

        Args:
            filepath: Path to file to edit
            old_string: Exact string to replace
            new_string: Replacement string
            replace_all: Replace all occurrences (default: False, must be unique)

        Returns:
            Dict with result or error
        """
        try:
            filepath = Path(filepath).expanduser().resolve()

            if not filepath.exists():
                return {"error": f"File not found: {filepath}"}

            # Read current content
            original_content = filepath.read_text()

            # Check if old_string exists
            if old_string not in original_content:
                return {
                    "error": "String not found in file",
                    "old_string": old_string[:100] + "..." if len(old_string) > 100 else old_string,
                    "filepath": str(filepath)
                }

            # Check uniqueness if not replace_all
            if not replace_all and original_content.count(old_string) > 1:
                return {
                    "error": f"String appears {original_content.count(old_string)} times (not unique)",
                    "suggestion": "Provide more context or use replace_all=True",
                    "old_string": old_string[:100]
                }

            # Create backup if enabled
            if self.create_backups:
                backup_path = Path(str(filepath) + ".bak")
                backup_path.write_text(original_content)

            # Perform replacement
            if replace_all:
                new_content = original_content.replace(old_string, new_string)
                replacements = original_content.count(old_string)
            else:
                new_content = original_content.replace(old_string, new_string, 1)
                replacements = 1

            # Write new content
            filepath.write_text(new_content)

            # Track edit
            edit_record = {
                "filepath": str(filepath),
                "old_string": old_string[:100],
                "new_string": new_string[:100],
                "replacements": replacements,
                "backup": str(filepath) + ".bak" if self.create_backups else None
            }
            self.edits_made.append(edit_record)

            return {
                "status": "success",
                "filepath": str(filepath),
                "replacements": replacements,
                "backup_created": self.create_backups,
                "preview": new_content[max(0, new_content.find(new_string)-100):
                                       new_content.find(new_string)+len(new_string)+100]
            }

        except Exception as e:
            return {"error": str(e), "filepath": str(filepath)}

    def write_file(self, filepath: str, content: str, overwrite: bool = False) -> Dict:
        """
        Write new file (like Claude Code Write tool)

        Args:
            filepath: Path for new file
            content: File contents
            overwrite: Allow overwriting existing file

        Returns:
            Dict with result or error
        """
        try:
            filepath = Path(filepath).expanduser().resolve()

            # Check if file exists
            if filepath.exists() and not overwrite:
                return {
                    "error": "File already exists",
                    "filepath": str(filepath),
                    "suggestion": "Use overwrite=True to replace"
                }

            # Create parent directories if needed
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            filepath.write_text(content)

            return {
                "status": "created" if not filepath.exists() else "overwritten",
                "filepath": str(filepath),
                "size": len(content),
                "lines": content.count('\n') + 1
            }

        except Exception as e:
            return {"error": str(e), "filepath": str(filepath)}

    def append_to_file(self, filepath: str, content: str) -> Dict:
        """Append content to file"""
        try:
            filepath = Path(filepath).expanduser().resolve()

            with open(filepath, 'a') as f:
                f.write(content)

            return {
                "status": "appended",
                "filepath": str(filepath),
                "appended_bytes": len(content)
            }

        except Exception as e:
            return {"error": str(e), "filepath": str(filepath)}

    def undo_last_edit(self) -> Dict:
        """Undo last edit by restoring backup"""
        if not self.edits_made:
            return {"error": "No edits to undo"}

        try:
            last_edit = self.edits_made[-1]
            backup_path = last_edit.get('backup')

            if not backup_path or not Path(backup_path).exists():
                return {"error": "No backup found for last edit"}

            # Restore from backup
            original_path = backup_path.replace('.bak', '')
            Path(backup_path).rename(original_path)

            self.edits_made.pop()

            return {
                "status": "undone",
                "filepath": original_path,
                "edit_undone": last_edit
            }

        except Exception as e:
            return {"error": str(e)}

    def get_edit_history(self) -> List[Dict]:
        """Get list of all edits made"""
        return self.edits_made

# CLI
if __name__ == "__main__":
    import sys
    import json

    ops = EditOps()

    if len(sys.argv) < 2:
        print("EditOps - Usage:")
        print("  python3 edit_operations.py edit /path/file.py 'old' 'new'")
        print("  python3 edit_operations.py write /path/file.py 'content'")
        print("  python3 edit_operations.py append /path/file.py 'content'")
        print("  python3 edit_operations.py undo")
        print("  python3 edit_operations.py history")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "edit" and len(sys.argv) > 4:
        result = ops.edit_file(sys.argv[2], sys.argv[3], sys.argv[4])
        print(json.dumps(result, indent=2))

    elif cmd == "write" and len(sys.argv) > 3:
        result = ops.write_file(sys.argv[2], sys.argv[3])
        print(json.dumps(result, indent=2))

    elif cmd == "append" and len(sys.argv) > 3:
        result = ops.append_to_file(sys.argv[2], sys.argv[3])
        print(json.dumps(result, indent=2))

    elif cmd == "undo":
        result = ops.undo_last_edit()
        print(json.dumps(result, indent=2))

    elif cmd == "history":
        history = ops.get_edit_history()
        print(json.dumps(history, indent=2))

#!/usr/bin/env python3
"""
Context Manager for Local Claude Code
Tracks project state, conversation history, and execution context
"""

import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class FileContext:
    """Context for a single file"""
    path: str
    first_accessed: float
    last_accessed: float
    access_count: int = 0
    content_hash: Optional[str] = None
    was_edited: bool = False
    edit_count: int = 0
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    lines: int = 0


@dataclass
class ConversationTurn:
    """Single turn in conversation"""
    timestamp: float
    user_message: str
    assistant_response: str
    task_type: str  # 'read', 'edit', 'analyze', 'execute', etc.
    files_involved: List[str]
    success: bool
    model_used: Optional[str] = None


class ContextManager:
    """
    Manages execution context and project state

    Features:
    - File access tracking
    - Edit history
    - Conversation memory
    - Project structure awareness
    - Cross-file relationship tracking
    - Session persistence
    """

    def __init__(self, workspace_root: str = ".", session_id: Optional[str] = None):
        """
        Initialize context manager

        Args:
            workspace_root: Project root directory
            session_id: Unique session identifier
        """
        self.workspace_root = Path(workspace_root).resolve()
        self.session_id = session_id or self._generate_session_id()

        # Context tracking
        self.files_accessed: Dict[str, FileContext] = {}
        self.conversation_history: List[ConversationTurn] = []
        self.edit_history: List[Dict] = []
        self.execution_log: List[Dict] = []

        # Project structure
        self.project_structure: Dict[str, Any] = {}
        self.file_relationships: Dict[str, List[str]] = defaultdict(list)

        # Session metadata
        self.session_start = time.time()
        self.total_edits = 0
        self.total_reads = 0
        self.total_executions = 0

        # Persistence
        self.session_file = self.workspace_root / ".local_claude_code" / f"session_{self.session_id}.json"
        self.session_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"ContextManager initialized (session: {self.session_id})")

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = str(time.time())
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]

    def add_file_access(
        self,
        filepath: str,
        content: Optional[str] = None,
        operation: str = "read"
    ) -> FileContext:
        """
        Record file access

        Args:
            filepath: File path
            content: File content (if available)
            operation: Operation type (read, edit, write)

        Returns:
            FileContext for the file
        """
        filepath = str(Path(filepath).resolve())

        if filepath not in self.files_accessed:
            # New file access
            ctx = FileContext(
                path=filepath,
                first_accessed=time.time(),
                last_accessed=time.time(),
                access_count=1
            )

            if content:
                ctx.content_hash = hashlib.md5(content.encode()).hexdigest()
                ctx.lines = content.count('\n') + 1

                # Extract structure if Python
                if filepath.endswith('.py'):
                    self._extract_python_structure(content, ctx)

            self.files_accessed[filepath] = ctx
        else:
            # Update existing
            ctx = self.files_accessed[filepath]
            ctx.last_accessed = time.time()
            ctx.access_count += 1

            if operation == "edit":
                ctx.was_edited = True
                ctx.edit_count += 1
                self.total_edits += 1

        self.total_reads += 1

        return ctx

    def _extract_python_structure(self, content: str, ctx: FileContext):
        """Extract Python code structure"""
        try:
            import ast

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    ctx.classes.append(node.name)
                elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                    ctx.functions.append(node.name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            ctx.imports.append(alias.name)
                    elif node.module:
                        ctx.imports.append(node.module)

        except SyntaxError:
            pass  # Invalid Python, skip structure extraction
        except Exception as e:
            logger.debug(f"Error extracting structure from {ctx.path}: {e}")

    def add_conversation_turn(
        self,
        user_message: str,
        assistant_response: str,
        task_type: str,
        files_involved: List[str],
        success: bool,
        model_used: Optional[str] = None
    ):
        """Record conversation turn"""
        turn = ConversationTurn(
            timestamp=time.time(),
            user_message=user_message,
            assistant_response=assistant_response,
            task_type=task_type,
            files_involved=files_involved,
            success=success,
            model_used=model_used
        )

        self.conversation_history.append(turn)

    def add_edit(self, filepath: str, old_content: str, new_content: str, description: str = ""):
        """Record file edit"""
        edit = {
            "timestamp": time.time(),
            "filepath": filepath,
            "old_hash": hashlib.md5(old_content.encode()).hexdigest(),
            "new_hash": hashlib.md5(new_content.encode()).hexdigest(),
            "description": description,
            "size_change": len(new_content) - len(old_content),
            "line_change": new_content.count('\n') - old_content.count('\n')
        }

        self.edit_history.append(edit)
        self.total_edits += 1

        # Update file context
        self.add_file_access(filepath, new_content, operation="edit")

    def add_execution(self, command: str, success: bool, output: str = "", error: str = ""):
        """Record command execution"""
        execution = {
            "timestamp": time.time(),
            "command": command,
            "success": success,
            "output": output[:500],  # Truncate output
            "error": error[:500]
        }

        self.execution_log.append(execution)
        self.total_executions += 1

    def get_project_context(self) -> str:
        """
        Build comprehensive project context string

        Returns:
            Human-readable context summary
        """
        ctx = []

        # Session info
        duration = time.time() - self.session_start
        ctx.append(f"Session: {self.session_id} (duration: {duration:.0f}s)")

        # Files accessed
        ctx.append(f"\nFiles accessed: {len(self.files_accessed)}")
        edited_files = [f for f, fc in self.files_accessed.items() if fc.was_edited]
        if edited_files:
            ctx.append(f"Files edited: {len(edited_files)}")
            for filepath in edited_files[:10]:  # Limit to 10
                fc = self.files_accessed[filepath]
                ctx.append(f"  - {Path(filepath).name} ({fc.edit_count} edits)")

        # Recent conversation
        ctx.append(f"\nConversation turns: {len(self.conversation_history)}")
        if self.conversation_history:
            recent = self.conversation_history[-3:]  # Last 3 turns
            for i, turn in enumerate(recent, 1):
                ctx.append(f"  {i}. [{turn.task_type}] {turn.user_message[:60]}...")

        # Execution stats
        ctx.append(f"\nExecutions: {self.total_executions}")
        ctx.append(f"Total edits: {self.total_edits}")
        ctx.append(f"Total reads: {self.total_reads}")

        return "\n".join(ctx)

    def get_file_context(self, filepath: str) -> Optional[FileContext]:
        """Get context for specific file"""
        filepath = str(Path(filepath).resolve())
        return self.files_accessed.get(filepath)

    def get_related_files(self, filepath: str) -> List[str]:
        """
        Get files related to given file

        Returns:
            List of related file paths
        """
        filepath = str(Path(filepath).resolve())

        # Check if we have relationships cached
        if filepath in self.file_relationships:
            return self.file_relationships[filepath]

        related = []

        # Same directory
        file_path = Path(filepath)
        same_dir = [str(f) for f in file_path.parent.glob('*.py') if f != file_path]
        related.extend(same_dir[:5])  # Limit to 5

        # Check imports
        file_ctx = self.get_file_context(filepath)
        if file_ctx and file_ctx.imports:
            for imp in file_ctx.imports:
                # Try to find the import
                import_path = self.workspace_root / imp.replace('.', '/') / '__init__.py'
                if import_path.exists():
                    related.append(str(import_path))

        # Cache relationships
        self.file_relationships[filepath] = related

        return related

    def get_conversation_context(self, last_n: int = 5) -> str:
        """Get recent conversation context"""
        if not self.conversation_history:
            return "No previous conversation"

        recent = self.conversation_history[-last_n:]

        context = []
        for i, turn in enumerate(recent, 1):
            context.append(f"Turn {i}:")
            context.append(f"  User: {turn.user_message[:100]}")
            context.append(f"  Task: {turn.task_type}")
            context.append(f"  Files: {', '.join(turn.files_involved) if turn.files_involved else 'None'}")
            context.append(f"  Result: {'✓ Success' if turn.success else '✗ Failed'}")

        return "\n".join(context)

    def get_edit_summary(self) -> Dict[str, Any]:
        """Get summary of all edits"""
        if not self.edit_history:
            return {"total": 0, "files": []}

        files_edited = defaultdict(int)
        total_size_change = 0
        total_line_change = 0

        for edit in self.edit_history:
            files_edited[edit['filepath']] += 1
            total_size_change += edit['size_change']
            total_line_change += edit['line_change']

        return {
            "total": len(self.edit_history),
            "files": dict(files_edited),
            "unique_files": len(files_edited),
            "size_change": total_size_change,
            "line_change": total_line_change
        }

    def save_session(self):
        """Save session to disk"""
        try:
            session_data = {
                "session_id": self.session_id,
                "workspace_root": str(self.workspace_root),
                "session_start": self.session_start,
                "duration": time.time() - self.session_start,
                "files_accessed": {
                    path: asdict(ctx) for path, ctx in self.files_accessed.items()
                },
                "conversation_history": [
                    asdict(turn) for turn in self.conversation_history
                ],
                "edit_history": self.edit_history,
                "execution_log": self.execution_log,
                "stats": {
                    "total_edits": self.total_edits,
                    "total_reads": self.total_reads,
                    "total_executions": self.total_executions,
                    "files_edited": len([f for f, fc in self.files_accessed.items() if fc.was_edited])
                }
            }

            with open(self.session_file, 'w') as f:
                json.dump(session_data, f, indent=2)

            logger.info(f"Session saved: {self.session_file}")

        except Exception as e:
            logger.error(f"Error saving session: {e}")

    def load_session(self, session_id: str) -> bool:
        """Load previous session"""
        try:
            session_file = self.workspace_root / ".local_claude_code" / f"session_{session_id}.json"

            if not session_file.exists():
                logger.warning(f"Session not found: {session_id}")
                return False

            with open(session_file, 'r') as f:
                session_data = json.load(f)

            # Restore state
            self.session_id = session_data['session_id']
            self.session_start = session_data['session_start']

            # Restore files
            for path, ctx_data in session_data['files_accessed'].items():
                self.files_accessed[path] = FileContext(**ctx_data)

            # Restore conversation
            for turn_data in session_data['conversation_history']:
                self.conversation_history.append(ConversationTurn(**turn_data))

            # Restore edits and executions
            self.edit_history = session_data['edit_history']
            self.execution_log = session_data['execution_log']

            # Restore stats
            stats = session_data.get('stats', {})
            self.total_edits = stats.get('total_edits', 0)
            self.total_reads = stats.get('total_reads', 0)
            self.total_executions = stats.get('total_executions', 0)

            logger.info(f"Session loaded: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error loading session: {e}")
            return False

    def get_session_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics"""
        return {
            "session_id": self.session_id,
            "duration_seconds": time.time() - self.session_start,
            "files_accessed": len(self.files_accessed),
            "files_edited": len([f for f, fc in self.files_accessed.items() if fc.was_edited]),
            "conversation_turns": len(self.conversation_history),
            "successful_turns": len([t for t in self.conversation_history if t.success]),
            "total_edits": self.total_edits,
            "total_reads": self.total_reads,
            "total_executions": self.total_executions,
            "success_rate": len([t for t in self.conversation_history if t.success]) / len(self.conversation_history) if self.conversation_history else 0.0
        }


def main():
    """Example usage"""
    print("=== Context Manager Demo ===\n")

    ctx_mgr = ContextManager()

    # Simulate file access
    print("1. Recording file access...")
    ctx_mgr.add_file_access("/home/user/project/server.py", content="def main():\n    pass\n")
    ctx_mgr.add_file_access("/home/user/project/utils.py", content="def helper():\n    pass\n")

    # Simulate edit
    print("2. Recording edit...")
    old_content = "def main():\n    pass\n"
    new_content = "def main():\n    print('Hello')\n    pass\n"
    ctx_mgr.add_edit("/home/user/project/server.py", old_content, new_content, "Added print statement")

    # Simulate conversation
    print("3. Recording conversation...")
    ctx_mgr.add_conversation_turn(
        user_message="Add logging to server.py",
        assistant_response="Added logging statements",
        task_type="edit",
        files_involved=["/home/user/project/server.py"],
        success=True,
        model_used="qwen2.5-coder"
    )

    # Get project context
    print("\n4. Project Context:")
    print(ctx_mgr.get_project_context())

    # Get stats
    print("\n5. Session Stats:")
    stats = ctx_mgr.get_session_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Save session
    print(f"\n6. Saving session...")
    ctx_mgr.save_session()
    print(f"   Saved to: {ctx_mgr.session_file}")

    print("\n✓ Context Manager ready!")


if __name__ == "__main__":
    main()

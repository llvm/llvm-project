#!/usr/bin/env python3
"""
ACE-FCA Abstract Interfaces Module
-----------------------------------
Provides abstract interfaces for external operations to decouple subagents
from direct dependencies on subprocess, file system, and human review mechanisms.

Addresses: Direct subprocess coupling, file I/O coupling, testability issues
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import subprocess
import os
import json
from enum import Enum


class ReviewStatus(Enum):
    """Status of a review request"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


@dataclass
class CommandResult:
    """Result of a command execution"""
    stdout: str
    stderr: str
    returncode: int
    success: bool

    @property
    def output(self) -> str:
        """Combined output (stdout + stderr)"""
        return f"{self.stdout}\n{self.stderr}".strip()


@dataclass
class ReviewRequest:
    """Request for human review"""
    title: str
    content: str
    context: Dict[str, Any]
    timeout_seconds: Optional[int] = None


@dataclass
class ReviewResult:
    """Result of a review request"""
    status: ReviewStatus
    feedback: Optional[str] = None
    approved: bool = False
    metadata: Optional[Dict[str, Any]] = None


# =============================================================================
# Abstract Interfaces
# =============================================================================

class FileSystemInterface(ABC):
    """Abstract interface for file system operations"""

    @abstractmethod
    def read_file(self, path: str) -> str:
        """Read file contents"""
        pass

    @abstractmethod
    def write_file(self, path: str, content: str) -> bool:
        """Write content to file"""
        pass

    @abstractmethod
    def file_exists(self, path: str) -> bool:
        """Check if file exists"""
        pass

    @abstractmethod
    def list_files(self, directory: str, pattern: Optional[str] = None) -> List[str]:
        """List files in directory, optionally filtered by pattern"""
        pass

    @abstractmethod
    def create_directory(self, path: str) -> bool:
        """Create directory (including parents)"""
        pass

    @abstractmethod
    def get_file_info(self, path: str) -> Dict[str, Any]:
        """Get file metadata (size, modified time, etc.)"""
        pass


class CommandExecutorInterface(ABC):
    """Abstract interface for command execution"""

    @abstractmethod
    def execute(
        self,
        command: str,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> CommandResult:
        """Execute a shell command"""
        pass

    @abstractmethod
    def execute_async(
        self,
        command: str,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None
    ) -> Any:
        """Execute command asynchronously (returns process handle)"""
        pass

    @abstractmethod
    def find_files(
        self,
        directory: str,
        pattern: Optional[str] = None,
        file_type: Optional[str] = None
    ) -> List[str]:
        """Find files matching criteria"""
        pass


class ReviewInterface(ABC):
    """Abstract interface for human review mechanism"""

    @abstractmethod
    def request_review(self, request: ReviewRequest) -> ReviewResult:
        """Request human review and wait for response"""
        pass

    @abstractmethod
    def is_enabled(self) -> bool:
        """Check if review mechanism is enabled"""
        pass

    @abstractmethod
    def set_auto_approve(self, enabled: bool):
        """Enable/disable auto-approval (for testing)"""
        pass


# =============================================================================
# Concrete Implementations (Production)
# =============================================================================

class StandardFileSystem(FileSystemInterface):
    """Standard file system implementation using os and pathlib"""

    def read_file(self, path: str) -> str:
        """Read file contents"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise IOError(f"Failed to read file {path}: {e}")

    def write_file(self, path: str, content: str) -> bool:
        """Write content to file"""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            raise IOError(f"Failed to write file {path}: {e}")

    def file_exists(self, path: str) -> bool:
        """Check if file exists"""
        return Path(path).exists()

    def list_files(self, directory: str, pattern: Optional[str] = None) -> List[str]:
        """List files in directory, optionally filtered by pattern"""
        dir_path = Path(directory)
        if not dir_path.exists():
            return []

        if pattern:
            return [str(p) for p in dir_path.glob(pattern)]
        else:
            return [str(p) for p in dir_path.iterdir() if p.is_file()]

    def create_directory(self, path: str) -> bool:
        """Create directory (including parents)"""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            raise IOError(f"Failed to create directory {path}: {e}")

    def get_file_info(self, path: str) -> Dict[str, Any]:
        """Get file metadata"""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")

        stat = p.stat()
        return {
            'size': stat.st_size,
            'modified': stat.st_mtime,
            'created': stat.st_ctime,
            'is_file': p.is_file(),
            'is_dir': p.is_dir(),
            'name': p.name,
            'extension': p.suffix
        }


class SubprocessCommandExecutor(CommandExecutorInterface):
    """Command executor using subprocess module"""

    def execute(
        self,
        command: str,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> CommandResult:
        """Execute a shell command"""
        try:
            # Merge environment variables
            exec_env = os.environ.copy()
            if env:
                exec_env.update(env)

            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                env=exec_env,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            return CommandResult(
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
                success=(result.returncode == 0)
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                returncode=-1,
                success=False
            )
        except Exception as e:
            return CommandResult(
                stdout="",
                stderr=f"Command execution failed: {e}",
                returncode=-1,
                success=False
            )

    def execute_async(
        self,
        command: str,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None
    ) -> subprocess.Popen:
        """Execute command asynchronously"""
        exec_env = os.environ.copy()
        if env:
            exec_env.update(env)

        return subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            env=exec_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    def find_files(
        self,
        directory: str,
        pattern: Optional[str] = None,
        file_type: Optional[str] = None
    ) -> List[str]:
        """Find files using find command"""
        cmd_parts = [f"find {directory}"]

        if file_type:
            cmd_parts.append(f"-type {file_type}")

        if pattern:
            cmd_parts.append(f"-name '{pattern}'")

        command = " ".join(cmd_parts)
        result = self.execute(command)

        if result.success:
            return [line.strip() for line in result.stdout.split('\n') if line.strip()]
        else:
            return []


class InteractiveReview(ReviewInterface):
    """Interactive human review implementation"""

    def __init__(self, auto_approve: bool = False):
        self._auto_approve = auto_approve
        self._enabled = True

    def request_review(self, request: ReviewRequest) -> ReviewResult:
        """Request human review and wait for response"""
        if not self._enabled:
            return ReviewResult(
                status=ReviewStatus.APPROVED,
                approved=True,
                feedback="Review disabled"
            )

        if self._auto_approve:
            return ReviewResult(
                status=ReviewStatus.APPROVED,
                approved=True,
                feedback="Auto-approved"
            )

        # Display review request
        print("\n" + "=" * 80)
        print(f"REVIEW REQUEST: {request.title}")
        print("=" * 80)
        print(request.content)
        print("\nContext:")
        print(json.dumps(request.context, indent=2))
        print("=" * 80)

        # Get user input
        while True:
            response = input("\nApprove? (y/n/comment): ").strip().lower()

            if response == 'y':
                return ReviewResult(
                    status=ReviewStatus.APPROVED,
                    approved=True
                )
            elif response == 'n':
                feedback = input("Rejection reason (optional): ").strip()
                return ReviewResult(
                    status=ReviewStatus.REJECTED,
                    approved=False,
                    feedback=feedback if feedback else None
                )
            elif response == 'comment':
                feedback = input("Your feedback: ").strip()
                return ReviewResult(
                    status=ReviewStatus.APPROVED,
                    approved=True,
                    feedback=feedback
                )
            else:
                print("Invalid response. Please enter 'y', 'n', or 'comment'")

    def is_enabled(self) -> bool:
        """Check if review is enabled"""
        return self._enabled

    def set_auto_approve(self, enabled: bool):
        """Enable/disable auto-approval"""
        self._auto_approve = enabled


# =============================================================================
# Mock Implementations (Testing)
# =============================================================================

class MockFileSystem(FileSystemInterface):
    """Mock file system for testing"""

    def __init__(self):
        self.files: Dict[str, str] = {}
        self.directories: set = set()

    def read_file(self, path: str) -> str:
        if path not in self.files:
            raise FileNotFoundError(f"File not found: {path}")
        return self.files[path]

    def write_file(self, path: str, content: str) -> bool:
        self.files[path] = content
        # Auto-create parent directory
        parent = str(Path(path).parent)
        self.directories.add(parent)
        return True

    def file_exists(self, path: str) -> bool:
        return path in self.files

    def list_files(self, directory: str, pattern: Optional[str] = None) -> List[str]:
        import fnmatch
        files = [p for p in self.files.keys() if str(Path(p).parent) == directory]
        if pattern:
            files = [f for f in files if fnmatch.fnmatch(Path(f).name, pattern)]
        return files

    def create_directory(self, path: str) -> bool:
        self.directories.add(path)
        return True

    def get_file_info(self, path: str) -> Dict[str, Any]:
        if path not in self.files:
            raise FileNotFoundError(f"File not found: {path}")

        return {
            'size': len(self.files[path]),
            'modified': 0,
            'created': 0,
            'is_file': True,
            'is_dir': False,
            'name': Path(path).name,
            'extension': Path(path).suffix
        }


class MockCommandExecutor(CommandExecutorInterface):
    """Mock command executor for testing"""

    def __init__(self):
        self.command_history: List[str] = []
        self.command_results: Dict[str, CommandResult] = {}

    def set_result(self, command: str, result: CommandResult):
        """Pre-configure result for a command"""
        self.command_results[command] = result

    def execute(
        self,
        command: str,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> CommandResult:
        self.command_history.append(command)

        # Return pre-configured result if available
        if command in self.command_results:
            return self.command_results[command]

        # Default success result
        return CommandResult(
            stdout=f"Mock output for: {command}",
            stderr="",
            returncode=0,
            success=True
        )

    def execute_async(
        self,
        command: str,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None
    ) -> Any:
        self.command_history.append(command)
        return None  # Mock process handle

    def find_files(
        self,
        directory: str,
        pattern: Optional[str] = None,
        file_type: Optional[str] = None
    ) -> List[str]:
        # Return mock file list
        return [f"{directory}/file1.py", f"{directory}/file2.py"]


class MockReview(ReviewInterface):
    """Mock review for testing"""

    def __init__(self, auto_approve: bool = True):
        self._auto_approve = auto_approve
        self._enabled = True
        self.review_history: List[ReviewRequest] = []

    def request_review(self, request: ReviewRequest) -> ReviewResult:
        self.review_history.append(request)

        if not self._enabled:
            return ReviewResult(
                status=ReviewStatus.APPROVED,
                approved=True,
                feedback="Review disabled"
            )

        if self._auto_approve:
            return ReviewResult(
                status=ReviewStatus.APPROVED,
                approved=True,
                feedback="Auto-approved (mock)"
            )
        else:
            return ReviewResult(
                status=ReviewStatus.REJECTED,
                approved=False,
                feedback="Auto-rejected (mock)"
            )

    def is_enabled(self) -> bool:
        return self._enabled

    def set_auto_approve(self, enabled: bool):
        self._auto_approve = enabled


# =============================================================================
# Factory Functions
# =============================================================================

def create_production_interfaces() -> Tuple[FileSystemInterface, CommandExecutorInterface, ReviewInterface]:
    """Create production interfaces"""
    return (
        StandardFileSystem(),
        SubprocessCommandExecutor(),
        InteractiveReview()
    )


def create_test_interfaces(
    auto_approve: bool = True
) -> Tuple[MockFileSystem, MockCommandExecutor, MockReview]:
    """Create test/mock interfaces"""
    return (
        MockFileSystem(),
        MockCommandExecutor(),
        MockReview(auto_approve=auto_approve)
    )


if __name__ == "__main__":
    # Example usage
    print("Production Interfaces:")
    fs, cmd, review = create_production_interfaces()

    # Test file system
    print(f"File exists: {fs.file_exists('/tmp/test.txt')}")

    # Test command executor
    result = cmd.execute("echo 'Hello World'")
    print(f"Command result: {result.stdout.strip()}")

    print("\n" + "=" * 80)
    print("Mock Interfaces:")
    mock_fs, mock_cmd, mock_review = create_test_interfaces()

    # Test mock file system
    mock_fs.write_file("/mock/test.txt", "Mock content")
    print(f"Mock file content: {mock_fs.read_file('/mock/test.txt')}")

    # Test mock command
    mock_result = mock_cmd.execute("ls -la")
    print(f"Mock command result: {mock_result.stdout}")
    print(f"Command history: {mock_cmd.command_history}")

    # Test mock review
    review_req = ReviewRequest(
        title="Test Review",
        content="Testing review system",
        context={"test": True}
    )
    review_result = mock_review.request_review(review_req)
    print(f"Mock review result: {review_result.status.value}, approved={review_result.approved}")

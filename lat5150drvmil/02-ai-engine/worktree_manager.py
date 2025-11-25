#!/usr/bin/env python3
"""
Git Worktree Manager for Parallel Development

Enables multiple feature branches to be active simultaneously without conflicts.
Inspired by HumanLayer's worktree support for parallel agent execution.

Key Features:
- Create/delete git worktrees automatically
- Associate worktrees with parallel agent tasks
- Isolated working directories for each task
- Automatic cleanup on task completion
- Branch management integration

Based on: HumanLayer/CodeLayer worktree management
"""

import os
import subprocess
import shutil
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json


@dataclass
class Worktree:
    """Represents a git worktree"""
    worktree_id: str
    path: Path
    branch: str
    task_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    is_locked: bool = False
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "worktree_id": self.worktree_id,
            "path": str(self.path),
            "branch": self.branch,
            "task_id": self.task_id,
            "created_at": self.created_at.isoformat(),
            "is_locked": self.is_locked,
            "metadata": self.metadata
        }


class WorktreeManager:
    """
    Git Worktree Manager - Enable parallel development threads

    Manages git worktrees for parallel agent execution:
    - Each agent gets an isolated working directory
    - No conflicts between concurrent tasks
    - Automatic cleanup when tasks complete
    """

    def __init__(self,
                 repo_path: str = ".",
                 worktrees_base_dir: str = ".worktrees"):
        """
        Initialize worktree manager

        Args:
            repo_path: Path to git repository root
            worktrees_base_dir: Base directory for worktrees (relative to repo)
        """
        self.repo_path = Path(repo_path).resolve()
        self.worktrees_base_dir = self.repo_path / worktrees_base_dir

        # Ensure base directory exists
        self.worktrees_base_dir.mkdir(exist_ok=True)

        # Track active worktrees
        self.worktrees: Dict[str, Worktree] = {}
        self._worktree_counter = 0

        # Load existing worktrees
        self._discover_worktrees()

    def _run_git(self, args: List[str], cwd: Path = None) -> Tuple[bool, str, str]:
        """
        Run git command

        Args:
            args: Git command arguments
            cwd: Working directory (default: repo root)

        Returns:
            (success, stdout, stderr)
        """
        cwd = cwd or self.repo_path

        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30
            )

            return (
                result.returncode == 0,
                result.stdout.strip(),
                result.stderr.strip()
            )

        except subprocess.TimeoutExpired:
            return False, "", "Git command timed out"
        except Exception as e:
            return False, "", str(e)

    def _discover_worktrees(self):
        """Discover existing worktrees from git"""
        success, stdout, _ = self._run_git(["worktree", "list", "--porcelain"])

        if not success:
            return

        # Parse worktree list
        worktrees_data = []
        current = {}

        for line in stdout.split('\n'):
            if line.startswith('worktree '):
                if current:
                    worktrees_data.append(current)
                current = {'path': line.split(' ', 1)[1]}
            elif line.startswith('branch '):
                current['branch'] = line.split(' ', 1)[1].replace('refs/heads/', '')
            elif line.startswith('locked'):
                current['locked'] = True

        if current:
            worktrees_data.append(current)

        # Track worktrees in our base directory
        for wt_data in worktrees_data:
            wt_path = Path(wt_data['path'])

            # Only track worktrees in our managed directory
            if wt_path.parent == self.worktrees_base_dir:
                worktree_id = wt_path.name
                self.worktrees[worktree_id] = Worktree(
                    worktree_id=worktree_id,
                    path=wt_path,
                    branch=wt_data.get('branch', 'unknown'),
                    is_locked=wt_data.get('locked', False)
                )

    def create_worktree(self,
                       branch_name: Optional[str] = None,
                       base_branch: str = "main",
                       task_id: Optional[str] = None,
                       description: str = "") -> Optional[Worktree]:
        """
        Create a new git worktree for parallel development

        Args:
            branch_name: Name for new branch (auto-generated if None)
            base_branch: Branch to create from (default: main)
            task_id: Associated parallel task ID
            description: Description of the work

        Returns:
            Worktree object if successful, None otherwise
        """
        # Generate branch name if not provided
        if not branch_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            branch_name = f"parallel/{timestamp}_{self._worktree_counter}"
            self._worktree_counter += 1

        # Worktree directory name
        worktree_id = branch_name.replace('/', '_')
        worktree_path = self.worktrees_base_dir / worktree_id

        # Check if worktree already exists
        if worktree_path.exists():
            print(f"⚠️  Worktree path already exists: {worktree_path}")
            return None

        # Create worktree with new branch
        success, stdout, stderr = self._run_git([
            "worktree", "add",
            "-b", branch_name,
            str(worktree_path),
            base_branch
        ])

        if not success:
            print(f"❌ Failed to create worktree: {stderr}")
            return None

        # Create worktree object
        worktree = Worktree(
            worktree_id=worktree_id,
            path=worktree_path,
            branch=branch_name,
            task_id=task_id,
            metadata={"description": description}
        )

        self.worktrees[worktree_id] = worktree

        print(f"✅ Created worktree: {worktree_id} → {branch_name}")
        return worktree

    def remove_worktree(self, worktree_id: str, force: bool = False) -> bool:
        """
        Remove a git worktree

        Args:
            worktree_id: Worktree ID to remove
            force: Force removal even if worktree has uncommitted changes

        Returns:
            True if successful, False otherwise
        """
        if worktree_id not in self.worktrees:
            print(f"⚠️  Worktree not found: {worktree_id}")
            return False

        worktree = self.worktrees[worktree_id]

        # Remove worktree
        args = ["worktree", "remove", str(worktree.path)]
        if force:
            args.append("--force")

        success, stdout, stderr = self._run_git(args)

        if not success:
            print(f"❌ Failed to remove worktree: {stderr}")
            return False

        # Remove from tracking
        del self.worktrees[worktree_id]

        print(f"✅ Removed worktree: {worktree_id}")
        return True

    def get_worktree(self, worktree_id: str) -> Optional[Worktree]:
        """Get worktree by ID"""
        return self.worktrees.get(worktree_id)

    def get_worktree_by_task(self, task_id: str) -> Optional[Worktree]:
        """Get worktree associated with a parallel task"""
        for worktree in self.worktrees.values():
            if worktree.task_id == task_id:
                return worktree
        return None

    def list_worktrees(self) -> List[Worktree]:
        """List all managed worktrees"""
        return list(self.worktrees.values())

    def cleanup_stale_worktrees(self, days: int = 7) -> List[str]:
        """
        Clean up worktrees older than specified days

        Args:
            days: Age threshold in days

        Returns:
            List of removed worktree IDs
        """
        removed = []
        cutoff_time = datetime.now().timestamp() - (days * 86400)

        for worktree_id, worktree in list(self.worktrees.items()):
            if worktree.created_at.timestamp() < cutoff_time:
                if self.remove_worktree(worktree_id, force=True):
                    removed.append(worktree_id)

        return removed

    def get_worktree_status(self, worktree_id: str) -> Optional[Dict]:
        """
        Get git status for a worktree

        Args:
            worktree_id: Worktree ID

        Returns:
            Status dict with changes, branch, etc.
        """
        if worktree_id not in self.worktrees:
            return None

        worktree = self.worktrees[worktree_id]

        # Get git status
        success, stdout, stderr = self._run_git(
            ["status", "--porcelain"],
            cwd=worktree.path
        )

        if not success:
            return {"error": stderr}

        # Parse changes
        changes = []
        for line in stdout.split('\n'):
            if line.strip():
                changes.append(line)

        # Get current branch
        success, branch, _ = self._run_git(
            ["branch", "--show-current"],
            cwd=worktree.path
        )

        return {
            "worktree_id": worktree_id,
            "branch": branch if success else worktree.branch,
            "has_changes": len(changes) > 0,
            "changes_count": len(changes),
            "changes": changes[:10],  # First 10 changes
            "path": str(worktree.path)
        }

    def commit_worktree_changes(self,
                                worktree_id: str,
                                message: str,
                                add_all: bool = True) -> bool:
        """
        Commit changes in a worktree

        Args:
            worktree_id: Worktree ID
            message: Commit message
            add_all: Add all changes before committing

        Returns:
            True if successful
        """
        if worktree_id not in self.worktrees:
            return False

        worktree = self.worktrees[worktree_id]

        # Add changes if requested
        if add_all:
            success, _, stderr = self._run_git(
                ["add", "."],
                cwd=worktree.path
            )
            if not success:
                print(f"❌ Failed to add changes: {stderr}")
                return False

        # Commit
        success, stdout, stderr = self._run_git(
            ["commit", "-m", message],
            cwd=worktree.path
        )

        if not success:
            print(f"❌ Failed to commit: {stderr}")
            return False

        print(f"✅ Committed changes in {worktree_id}: {message}")
        return True

    def merge_worktree_to_main(self,
                               worktree_id: str,
                               target_branch: str = "main",
                               delete_after: bool = True) -> bool:
        """
        Merge worktree branch to target branch

        Args:
            worktree_id: Worktree ID
            target_branch: Target branch (default: main)
            delete_after: Delete worktree after merge

        Returns:
            True if successful
        """
        if worktree_id not in self.worktrees:
            return False

        worktree = self.worktrees[worktree_id]

        # Switch to target branch in main repo
        success, _, stderr = self._run_git(["checkout", target_branch])
        if not success:
            print(f"❌ Failed to checkout {target_branch}: {stderr}")
            return False

        # Merge worktree branch
        success, stdout, stderr = self._run_git(
            ["merge", "--no-ff", worktree.branch, "-m", f"Merge {worktree.branch}"]
        )

        if not success:
            print(f"❌ Failed to merge: {stderr}")
            return False

        print(f"✅ Merged {worktree.branch} → {target_branch}")

        # Delete worktree if requested
        if delete_after:
            self.remove_worktree(worktree_id)

        return True

    def get_stats(self) -> Dict:
        """Get worktree manager statistics"""
        return {
            "total_worktrees": len(self.worktrees),
            "with_tasks": sum(1 for w in self.worktrees.values() if w.task_id),
            "locked": sum(1 for w in self.worktrees.values() if w.is_locked),
            "base_directory": str(self.worktrees_base_dir),
            "worktrees": [w.to_dict() for w in self.worktrees.values()]
        }


# Example usage
if __name__ == "__main__":
    print("Git Worktree Manager - Parallel Development Support")
    print("=" * 70)

    # Initialize manager
    manager = WorktreeManager()

    print(f"\n📁 Base directory: {manager.worktrees_base_dir}")
    print(f"📋 Existing worktrees: {len(manager.worktrees)}\n")

    # Create worktree
    worktree = manager.create_worktree(
        base_branch="main",
        task_id="test_task_1",
        description="Test parallel development"
    )

    if worktree:
        print(f"\n📊 Worktree created:")
        print(f"   ID: {worktree.worktree_id}")
        print(f"   Path: {worktree.path}")
        print(f"   Branch: {worktree.branch}")

        # Get status
        status = manager.get_worktree_status(worktree.worktree_id)
        print(f"\n📈 Status:")
        print(f"   Branch: {status['branch']}")
        print(f"   Changes: {status['changes_count']}")

    # Show stats
    stats = manager.get_stats()
    print(f"\n📊 Stats:")
    print(f"   Total worktrees: {stats['total_worktrees']}")
    print(f"   With tasks: {stats['with_tasks']}")

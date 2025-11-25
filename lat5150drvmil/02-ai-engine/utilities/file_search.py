#!/usr/bin/env python3
"""
File Search Utility
-------------------
Dedicated utility for file discovery and search operations.

Extracted from ResearchAgent to improve modularity and reusability.
This utility handles all file system search operations and content filtering.
"""

import logging
from typing import List, Optional

from ace_interfaces import (
    FileSystemInterface,
    CommandExecutorInterface,
    StandardFileSystem,
    SubprocessCommandExecutor
)

logger = logging.getLogger(__name__)


class FileSearchUtility:
    """
    Utility for file discovery and search operations.

    Responsibilities:
    - Find files matching patterns
    - Filter files by content (query matching)
    - Deduplicate and limit results
    - Handle errors gracefully

    Uses dependency injection for file system and command execution,
    making it easily testable with mocks.
    """

    def __init__(
        self,
        filesystem: Optional[FileSystemInterface] = None,
        command_executor: Optional[CommandExecutorInterface] = None
    ):
        """
        Initialize file search utility with dependencies.

        Args:
            filesystem: File system interface (creates default if None)
            command_executor: Command executor interface (creates default if None)
        """
        self.filesystem = filesystem or StandardFileSystem()
        self.command_executor = command_executor or SubprocessCommandExecutor()

    def search_files(
        self,
        paths: List[str],
        patterns: List[str],
        query: Optional[str] = None,
        max_results: int = 50
    ) -> List[str]:
        """
        Search for files matching patterns and optionally containing query.

        Args:
            paths: List of directory paths to search
            patterns: List of file patterns (e.g., ["*.py", "*.js"])
            query: Optional text to search for in file contents
            max_results: Maximum number of files to return (default: 50)

        Returns:
            List of file paths matching criteria

        Examples:
            >>> searcher = FileSearchUtility()
            >>> files = searcher.search_files(["."], ["*.py"], "authentication")
            >>> print(f"Found {len(files)} Python files with 'authentication'")
        """
        files = []

        # 1. Find files matching patterns
        for path in paths:
            for pattern in patterns:
                try:
                    found_files = self.command_executor.find_files(
                        directory=path,
                        pattern=pattern,
                        file_type="f"
                    )
                    files.extend(found_files)
                except Exception as e:
                    logger.warning(f"Error searching {path} for {pattern}: {e}")

        # 2. Filter out empty strings and deduplicate
        files = list(set(f for f in files if f))

        # 3. If query provided, filter by content
        if query:
            matching_files = self._filter_by_content(files, query, max_results)
            return matching_files

        # 4. Return limited results
        return files[:max_results]

    def _filter_by_content(
        self,
        files: List[str],
        query: str,
        max_results: int
    ) -> List[str]:
        """
        Filter files by content matching query.

        Args:
            files: List of file paths to filter
            query: Text to search for in file contents
            max_results: Maximum number of files to return

        Returns:
            List of files containing query (case-insensitive)
        """
        matching_files = []
        query_lower = query.lower()

        for file in files:
            # Stop if we have enough results
            if len(matching_files) >= max_results:
                break

            try:
                content = self.filesystem.read_file(file)
                if query_lower in content.lower():
                    matching_files.append(file)
            except Exception as e:
                logger.debug(f"Could not read {file}: {e}")

        return matching_files

    def search_in_file(self, file_path: str, query: str) -> bool:
        """
        Check if a single file contains the query.

        Args:
            file_path: Path to file
            query: Text to search for

        Returns:
            True if file contains query (case-insensitive), False otherwise
        """
        try:
            content = self.filesystem.read_file(file_path)
            return query.lower() in content.lower()
        except Exception as e:
            logger.debug(f"Could not read {file_path}: {e}")
            return False

    def find_files_by_pattern(
        self,
        directory: str,
        pattern: str,
        recursive: bool = True
    ) -> List[str]:
        """
        Find files matching a pattern in a directory.

        Args:
            directory: Directory to search
            pattern: File pattern (e.g., "*.py", "test_*.js")
            recursive: Search subdirectories (default: True)

        Returns:
            List of file paths matching pattern
        """
        try:
            found_files = self.command_executor.find_files(
                directory=directory,
                pattern=pattern,
                file_type="f"
            )
            # Filter out empty strings
            return [f for f in found_files if f]
        except Exception as e:
            logger.warning(f"Error finding files in {directory} with pattern {pattern}: {e}")
            return []

    def get_file_count(self, paths: List[str], patterns: List[str]) -> int:
        """
        Get count of files matching patterns without reading content.

        Args:
            paths: List of directory paths
            patterns: List of file patterns

        Returns:
            Total count of matching files
        """
        files = self.search_files(paths, patterns, query=None, max_results=10000)
        return len(files)


# Example usage and testing
if __name__ == "__main__":
    import sys

    print("File Search Utility - Test Mode")
    print("=" * 60)

    # Create utility
    searcher = FileSearchUtility()

    # Test 1: Find all Python files
    print("\nTest 1: Find all Python files in current directory")
    py_files = searcher.search_files(["."], ["*.py"], max_results=10)
    print(f"Found {len(py_files)} Python files:")
    for f in py_files[:5]:
        print(f"  - {f}")

    # Test 2: Search for files with specific content
    if len(sys.argv) > 1:
        query = sys.argv[1]
        print(f"\nTest 2: Find Python files containing '{query}'")
        matching = searcher.search_files(["."], ["*.py"], query=query, max_results=10)
        print(f"Found {len(matching)} matching files:")
        for f in matching[:5]:
            print(f"  - {f}")

    # Test 3: Count files
    print("\nTest 3: Count all Python and Markdown files")
    count = searcher.get_file_count(["."], ["*.py", "*.md"])
    print(f"Total: {count} files")

    print("\n✓ All tests completed")

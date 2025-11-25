#!/usr/bin/env python3
"""
Code Analyzer Utility
---------------------
Dedicated utility for code structure and pattern analysis.

Extracted from ResearchAgent to improve modularity and reusability.
This utility handles architectural analysis and pattern detection.
"""

import os
import logging
from typing import List, Dict, Optional, Set

from ace_interfaces import (
    FileSystemInterface,
    StandardFileSystem
)

logger = logging.getLogger(__name__)


class CodeAnalyzer:
    """
    Utility for analyzing code structure and patterns.

    Responsibilities:
    - Analyze project architecture from file structure
    - Identify common patterns (OOP, async, decorators, etc.)
    - Extract directory structure insights
    - Detect technology stack indicators

    Uses dependency injection for file system access,
    making it easily testable with mocks.
    """

    def __init__(
        self,
        filesystem: Optional[FileSystemInterface] = None
    ):
        """
        Initialize code analyzer with dependencies.

        Args:
            filesystem: File system interface (creates default if None)
        """
        self.filesystem = filesystem or StandardFileSystem()

    def analyze_architecture(
        self,
        files: List[str],
        max_dirs: int = 10
    ) -> str:
        """
        Analyze architecture from file structure.

        Examines file paths to identify:
        - Directory structure and organization
        - Presence of API layers
        - Test coverage indicators
        - Configuration file patterns
        - Common architectural patterns

        Args:
            files: List of file paths to analyze
            max_dirs: Maximum number of directories to include in structure (default: 10)

        Returns:
            Human-readable architecture description

        Examples:
            >>> analyzer = CodeAnalyzer()
            >>> files = ["src/api/routes.py", "src/api/models.py", "tests/test_api.py"]
            >>> architecture = analyzer.analyze_architecture(files)
            >>> print(architecture)
            - Structure: src/api, tests
            - API layer present
            - Test coverage exists
        """
        if not files:
            return "No files found for analysis"

        # 1. Extract directory structure
        dirs = set(os.path.dirname(f) for f in files)
        dir_structure = sorted(dirs)[:max_dirs]

        # 2. Identify common architectural patterns
        architecture_lines = []
        architecture_lines.append("- Structure: " + ", ".join(dir_structure))

        # 3. Check for specific layers/patterns
        patterns = self._detect_architectural_patterns(files)
        architecture_lines.extend(patterns)

        return "\n".join(architecture_lines)

    def _detect_architectural_patterns(self, files: List[str]) -> List[str]:
        """
        Detect architectural patterns from file paths and names.

        Args:
            files: List of file paths

        Returns:
            List of detected patterns
        """
        patterns = []

        # Check for common directory/file patterns
        file_lower = [f.lower() for f in files]

        # API layer
        if any('api' in f or 'routes' in f or 'endpoints' in f for f in file_lower):
            patterns.append("- API layer present")

        # Test coverage
        if any('test' in f or 'spec' in f for f in file_lower):
            patterns.append("- Test coverage exists")

        # Configuration
        if any('config' in f or 'settings' in f for f in file_lower):
            patterns.append("- Configuration files found")

        # Database layer
        if any('models' in f or 'schema' in f or 'database' in f or 'db' in f for f in file_lower):
            patterns.append("- Database/ORM layer detected")

        # Services/Business logic
        if any('service' in f or 'business' in f or 'logic' in f for f in file_lower):
            patterns.append("- Service/business logic layer found")

        # Frontend indicators
        if any('component' in f or 'view' in f or 'template' in f for f in file_lower):
            patterns.append("- Frontend/UI components present")

        # Utilities
        if any('util' in f or 'helper' in f or 'common' in f for f in file_lower):
            patterns.append("- Utility/helper modules found")

        # Docker/Containerization
        if any('docker' in f or 'container' in f for f in file_lower):
            patterns.append("- Containerization setup (Docker)")

        # CI/CD
        if any('.github' in f or 'jenkins' in f or 'gitlab-ci' in f or '.circleci' in f for f in file_lower):
            patterns.append("- CI/CD pipeline configured")

        return patterns

    def find_patterns(
        self,
        files: List[str],
        query: Optional[str] = None,
        max_files_to_sample: int = 5,
        max_patterns: int = 10
    ) -> str:
        """
        Find implementation patterns in files.

        Analyzes file contents to detect:
        - Object-oriented patterns (classes)
        - Async/await patterns
        - Decorator usage
        - Functional programming patterns
        - Error handling patterns

        Args:
            files: List of file paths to analyze
            query: Optional query context (not currently used but reserved for future)
            max_files_to_sample: Maximum number of files to read (default: 5)
            max_patterns: Maximum patterns to return (default: 10)

        Returns:
            Human-readable description of patterns found

        Examples:
            >>> analyzer = CodeAnalyzer()
            >>> patterns = analyzer.find_patterns(["api.py", "models.py"])
            >>> print(patterns)
            - OOP patterns in api.py
            - Async patterns in api.py
            - Decorators in api.py
        """
        patterns = []

        # Sample first few files for pattern analysis
        for file in files[:max_files_to_sample]:
            try:
                # Read file content (first 2000 chars for efficiency)
                content = self.filesystem.read_file(file)
                content_sample = content[:2000]

                # Detect patterns in this file
                file_patterns = self._detect_code_patterns(file, content_sample)
                patterns.extend(file_patterns)

            except Exception as e:
                logger.debug(f"Could not analyze {file}: {e}")

        # Return top patterns or "no patterns" message
        if patterns:
            return "\n".join(patterns[:max_patterns])
        else:
            return "No clear patterns identified"

    def _detect_code_patterns(self, file_path: str, content: str) -> List[str]:
        """
        Detect code patterns in file content.

        Args:
            file_path: Path to file (for reporting)
            content: File content to analyze

        Returns:
            List of pattern descriptions
        """
        patterns = []
        filename = os.path.basename(file_path)

        # Object-Oriented Programming
        if 'class ' in content:
            patterns.append(f"- OOP patterns in {filename}")

        # Async/Await
        if 'async ' in content or 'await ' in content:
            patterns.append(f"- Async patterns in {filename}")

        # Decorators
        if '@' in content and 'def ' in content:
            patterns.append(f"- Decorators in {filename}")

        # Functional programming (lambda, map, filter, reduce)
        if any(keyword in content for keyword in ['lambda ', 'map(', 'filter(', 'reduce(']):
            patterns.append(f"- Functional programming in {filename}")

        # Type hints
        if '->' in content or ': ' in content:
            # Simple heuristic: check for type annotation syntax
            if any(t in content for t in ['List[', 'Dict[', 'Optional[', 'Tuple[', 'Set[']):
                patterns.append(f"- Type hints in {filename}")

        # Error handling
        if 'try:' in content and 'except' in content:
            patterns.append(f"- Error handling in {filename}")

        # Context managers
        if 'with ' in content:
            patterns.append(f"- Context managers in {filename}")

        # Dataclasses
        if '@dataclass' in content:
            patterns.append(f"- Dataclasses in {filename}")

        # Abstract base classes
        if 'ABC' in content or '@abstractmethod' in content:
            patterns.append(f"- Abstract interfaces in {filename}")

        # Dependency injection indicators
        if 'Optional[' in content and '__init__' in content:
            patterns.append(f"- Dependency injection pattern in {filename}")

        return patterns

    def get_technology_stack(self, files: List[str]) -> Dict[str, bool]:
        """
        Identify technology stack from file extensions and content.

        Args:
            files: List of file paths

        Returns:
            Dictionary mapping technology to presence boolean

        Examples:
            >>> analyzer = CodeAnalyzer()
            >>> stack = analyzer.get_technology_stack(["api.py", "app.js", "test.ts"])
            >>> print(stack)
            {'python': True, 'javascript': True, 'typescript': True}
        """
        stack = {
            'python': False,
            'javascript': False,
            'typescript': False,
            'go': False,
            'rust': False,
            'java': False,
            'csharp': False,
            'ruby': False,
            'php': False
        }

        # Check file extensions
        for file in files:
            ext = os.path.splitext(file)[1].lower()

            if ext == '.py':
                stack['python'] = True
            elif ext in ['.js', '.jsx']:
                stack['javascript'] = True
            elif ext in ['.ts', '.tsx']:
                stack['typescript'] = True
            elif ext == '.go':
                stack['go'] = True
            elif ext == '.rs':
                stack['rust'] = True
            elif ext == '.java':
                stack['java'] = True
            elif ext == '.cs':
                stack['csharp'] = True
            elif ext == '.rb':
                stack['ruby'] = True
            elif ext == '.php':
                stack['php'] = True

        return stack

    def analyze_file_complexity(self, file_path: str) -> Dict[str, int]:
        """
        Analyze complexity metrics for a single file.

        Args:
            file_path: Path to file to analyze

        Returns:
            Dictionary with complexity metrics (lines, functions, classes, etc.)
        """
        try:
            content = self.filesystem.read_file(file_path)

            metrics = {
                'lines': len(content.splitlines()),
                'functions': content.count('def '),
                'classes': content.count('class '),
                'comments': content.count('#'),
                'docstrings': content.count('"""') + content.count("'''"),
                'imports': content.count('import ') + content.count('from ')
            }

            return metrics

        except Exception as e:
            logger.warning(f"Could not analyze complexity of {file_path}: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    print("Code Analyzer Utility - Test Mode")
    print("=" * 60)

    # Create analyzer
    analyzer = CodeAnalyzer()

    # Test 1: Analyze architecture
    print("\nTest 1: Analyze architecture from sample files")
    sample_files = [
        "src/api/routes.py",
        "src/api/models.py",
        "src/services/auth.py",
        "tests/test_api.py",
        "config/settings.py"
    ]
    architecture = analyzer.analyze_architecture(sample_files)
    print("Architecture:")
    print(architecture)

    # Test 2: Detect technology stack
    print("\nTest 2: Detect technology stack")
    stack = analyzer.get_technology_stack(sample_files)
    print("Technology Stack:")
    for tech, present in stack.items():
        if present:
            print(f"  ✓ {tech}")

    # Test 3: Analyze actual file if it exists
    print("\nTest 3: Analyze real file patterns")
    import glob
    py_files = glob.glob("*.py")[:3]
    if py_files:
        patterns = analyzer.find_patterns(py_files)
        print("Patterns found:")
        print(patterns)
    else:
        print("No Python files found in current directory")

    print("\n✓ All tests completed")

"""
Utility Modules for ACE-FCA Subagents
-------------------------------------
Reusable, testable utility classes extracted from subagents.

Modules:
- file_search: File discovery and search utilities
- code_analyzer: Code structure and pattern analysis

Usage:
    from utilities.file_search import FileSearchUtility
    from utilities.code_analyzer import CodeAnalyzer

    searcher = FileSearchUtility(filesystem, command_executor)
    files = searcher.search_files(["."], ["*.py"], "authentication")

    analyzer = CodeAnalyzer(filesystem)
    architecture = analyzer.analyze_architecture(files)
"""

from .file_search import FileSearchUtility
from .code_analyzer import CodeAnalyzer

__all__ = [
    'FileSearchUtility',
    'CodeAnalyzer'
]

__version__ = '1.0.0'

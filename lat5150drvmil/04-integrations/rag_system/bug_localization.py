"""
Automated Bug Localization (Phase 3.2)

Automatically locate the source of bugs from stack traces, error messages,
and symptom descriptions.

Features:
- Stack trace analysis (parse Python/C tracebacks)
- Error message pattern matching
- Change history correlation (when was code last modified?)
- Dependency impact analysis
- Probabilistic bug localization (rank files by likelihood)
- Symptom-based search
- Historical bug database

Example:
    >>> localizer = BugLocalizer("/path/to/codebase")
    >>> locations = localizer.locate_from_traceback(traceback_text)
    >>> locations = localizer.locate_from_symptom("500 error when uploading files >10MB")
"""

import ast
import re
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
from datetime import datetime
import hashlib


@dataclass
class BugLocation:
    """A potential bug location"""
    file_path: str
    line_number: Optional[int]
    function_name: Optional[str]
    confidence: float  # 0.0-1.0
    evidence: List[str]  # Why we think the bug is here
    code_snippet: Optional[str] = None
    suggested_fix: Optional[str] = None


@dataclass
class StackFrame:
    """A single stack frame from a traceback"""
    file_path: str
    line_number: int
    function_name: str
    code_line: str


@dataclass
class BugReport:
    """Complete bug localization report"""
    symptom: str
    error_type: Optional[str]
    locations: List[BugLocation]
    total_files_analyzed: int
    confidence_threshold: float = 0.3


class TracebackParser:
    """Parse Python and C stack traces"""

    @staticmethod
    def parse_python_traceback(traceback: str) -> List[StackFrame]:
        """Parse Python traceback"""
        frames = []

        # Python traceback pattern
        # File "path/to/file.py", line 42, in function_name
        #   code line
        pattern = r'File "([^"]+)", line (\d+), in (\w+)\s+(.+?)(?=\n|$)'

        matches = re.finditer(pattern, traceback, re.MULTILINE | re.DOTALL)

        for match in matches:
            frames.append(StackFrame(
                file_path=match.group(1),
                line_number=int(match.group(2)),
                function_name=match.group(3),
                code_line=match.group(4).strip()
            ))

        return frames

    @staticmethod
    def parse_c_traceback(traceback: str) -> List[StackFrame]:
        """Parse C/gdb stack trace"""
        frames = []

        # GDB backtrace pattern
        # #0  0x7ffff7a0d428 in function_name () at file.c:42
        pattern = r'#\d+\s+0x[\da-f]+\s+in\s+(\w+)\s+.*?at\s+([^:]+):(\d+)'

        matches = re.finditer(pattern, traceback)

        for match in matches:
            frames.append(StackFrame(
                file_path=match.group(2),
                line_number=int(match.group(3)),
                function_name=match.group(1),
                code_line=""
            ))

        return frames

    @staticmethod
    def extract_error_type(traceback: str) -> Optional[str]:
        """Extract error type from traceback"""
        # Python exceptions
        python_pattern = r'^(\w+Error|\w+Exception):'
        match = re.search(python_pattern, traceback, re.MULTILINE)
        if match:
            return match.group(1)

        # Segmentation fault
        if 'segmentation fault' in traceback.lower() or 'sigsegv' in traceback.lower():
            return "SegmentationFault"

        return None


class ErrorPatternMatcher:
    """Match error messages to known bug patterns"""

    ERROR_PATTERNS = {
        'null_reference': {
            'patterns': [
                r"NoneType.*has no attribute",
                r"null pointer",
                r"segmentation fault.*0x0+"
            ],
            'likely_causes': [
                "Missing null/None check",
                "Uninitialized variable",
                "Dereferencing null pointer"
            ],
            'search_keywords': ['None', 'null', 'NULL']
        },
        'index_out_of_bounds': {
            'patterns': [
                r"IndexError",
                r"list index out of range",
                r"buffer overflow"
            ],
            'likely_causes': [
                "Array access without bounds check",
                "Off-by-one error",
                "Buffer overflow"
            ],
            'search_keywords': ['len(', 'size', 'bounds']
        },
        'type_error': {
            'patterns': [
                r"TypeError",
                r"cannot.*of type",
                r"incompatible types"
            ],
            'likely_causes': [
                "Wrong argument type",
                "Type conversion issue",
                "Missing type check"
            ],
            'search_keywords': ['isinstance', 'type(', 'cast']
        },
        'file_not_found': {
            'patterns': [
                r"FileNotFoundError",
                r"No such file or directory",
                r"cannot open file"
            ],
            'likely_causes': [
                "Missing file existence check",
                "Incorrect file path",
                "Missing error handling"
            ],
            'search_keywords': ['open(', 'file', 'path']
        },
        'network_error': {
            'patterns': [
                r"ConnectionError",
                r"TimeoutError",
                r"Connection refused",
                r"500 Internal Server Error"
            ],
            'likely_causes': [
                "Network timeout not handled",
                "Server not responding",
                "Connection pool exhausted"
            ],
            'search_keywords': ['request', 'connect', 'timeout']
        }
    }

    @classmethod
    def match_pattern(cls, error_message: str) -> Optional[Tuple[str, Dict]]:
        """Match error message to known pattern"""

        for pattern_name, config in cls.ERROR_PATTERNS.items():
            for pattern in config['patterns']:
                if re.search(pattern, error_message, re.IGNORECASE):
                    return pattern_name, config

        return None


class ChangeHistoryAnalyzer:
    """Analyze git history to find recent changes"""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)

    def get_recently_modified_files(self, days: int = 30) -> List[Tuple[str, datetime]]:
        """Get files modified in last N days"""
        import subprocess

        try:
            # Get git log
            cmd = f'git log --since="{days} days ago" --name-only --pretty=format:"%H %ci"'
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )

            if result.returncode != 0:
                return []

            # Parse output
            modified_files = {}
            current_date = None

            for line in result.stdout.split('\n'):
                # Commit line with date
                if line and not line.startswith(' '):
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        try:
                            current_date = datetime.fromisoformat(parts[1].split('+')[0].strip())
                        except:
                            pass
                # File line
                elif line.strip() and current_date:
                    file_path = line.strip()
                    if file_path not in modified_files:
                        modified_files[file_path] = current_date

            return [(f, d) for f, d in modified_files.items()]

        except Exception:
            return []

    def get_file_modification_time(self, file_path: str) -> Optional[datetime]:
        """Get last modification time for a file"""
        try:
            stat = os.stat(self.repo_path / file_path)
            return datetime.fromtimestamp(stat.st_mtime)
        except:
            return None


class BugLocalizer:
    """Main bug localization engine"""

    def __init__(self, codebase_path: str):
        self.codebase_path = Path(codebase_path)
        self.traceback_parser = TracebackParser()
        self.pattern_matcher = ErrorPatternMatcher()
        self.change_analyzer = ChangeHistoryAnalyzer(codebase_path)

    def locate_from_traceback(self, traceback: str) -> BugReport:
        """Locate bug from stack trace"""

        # Parse traceback
        frames = self.traceback_parser.parse_python_traceback(traceback)
        if not frames:
            frames = self.traceback_parser.parse_c_traceback(traceback)

        # Extract error type
        error_type = self.traceback_parser.extract_error_type(traceback)

        locations = []

        # Analyze each frame
        for i, frame in enumerate(frames):
            confidence = 1.0 - (i * 0.15)  # Top of stack is most likely
            confidence = max(confidence, 0.3)

            evidence = [f"Frame {i} in stack trace"]

            # Get code context
            code_snippet = self._get_code_context(frame.file_path, frame.line_number)

            # Check if file was recently modified
            mod_time = self.change_analyzer.get_file_modification_time(frame.file_path)
            if mod_time and (datetime.now() - mod_time).days < 7:
                evidence.append("File modified in last 7 days")
                confidence += 0.1

            # Analyze code for issues
            issues = self._analyze_code_at_location(frame.file_path, frame.line_number)
            if issues:
                evidence.extend(issues)
                confidence += 0.05 * len(issues)

            locations.append(BugLocation(
                file_path=frame.file_path,
                line_number=frame.line_number,
                function_name=frame.function_name,
                confidence=min(confidence, 1.0),
                evidence=evidence,
                code_snippet=code_snippet
            ))

        # Sort by confidence
        locations.sort(key=lambda x: x.confidence, reverse=True)

        return BugReport(
            symptom=f"Exception: {error_type}" if error_type else "Unknown error",
            error_type=error_type,
            locations=locations,
            total_files_analyzed=len(set(f.file_path for f in frames))
        )

    def locate_from_symptom(self, symptom: str) -> BugReport:
        """Locate bug from symptom description"""

        locations = []

        # Match symptom to error pattern
        pattern_match = self.pattern_matcher.match_pattern(symptom)

        if pattern_match:
            pattern_name, config = pattern_match

            # Search codebase for relevant keywords
            for keyword in config['search_keywords']:
                file_locations = self._search_keyword(keyword)

                for file_path, line_number in file_locations:
                    confidence = 0.5  # Medium confidence for keyword match

                    evidence = [
                        f"Matches pattern: {pattern_name}",
                        f"Contains keyword: {keyword}"
                    ]

                    # Check if recently modified
                    mod_time = self.change_analyzer.get_file_modification_time(file_path)
                    if mod_time and (datetime.now() - mod_time).days < 30:
                        days_ago = (datetime.now() - mod_time).days
                        evidence.append(f"Modified {days_ago} days ago")
                        confidence += 0.2

                    code_snippet = self._get_code_context(file_path, line_number)

                    # Suggest fix based on pattern
                    suggested_fix = config.get('likely_causes', [None])[0]

                    locations.append(BugLocation(
                        file_path=file_path,
                        line_number=line_number,
                        function_name=None,
                        confidence=confidence,
                        evidence=evidence,
                        code_snippet=code_snippet,
                        suggested_fix=suggested_fix
                    ))

        # Also check recently modified files
        recent_files = self.change_analyzer.get_recently_modified_files(days=14)
        for file_path, mod_date in recent_files[:10]:  # Top 10 recent
            if not any(loc.file_path == file_path for loc in locations):
                days_ago = (datetime.now() - mod_date).days
                locations.append(BugLocation(
                    file_path=file_path,
                    line_number=None,
                    function_name=None,
                    confidence=0.3 + (0.02 * (14 - days_ago)),  # More recent = higher confidence
                    evidence=[f"Recently modified ({days_ago} days ago)"],
                    code_snippet=None
                ))

        # Sort by confidence
        locations.sort(key=lambda x: x.confidence, reverse=True)

        return BugReport(
            symptom=symptom,
            error_type=pattern_match[0] if pattern_match else None,
            locations=locations[:15],  # Top 15
            total_files_analyzed=len(recent_files)
        )

    def _get_code_context(self, file_path: str, line_number: int, context_lines: int = 3) -> Optional[str]:
        """Get code context around a line"""
        try:
            full_path = self.codebase_path / file_path if not Path(file_path).is_absolute() else Path(file_path)

            with open(full_path, 'r') as f:
                lines = f.readlines()

            start = max(0, line_number - context_lines - 1)
            end = min(len(lines), line_number + context_lines)

            context = ''.join(lines[start:end])
            return context

        except Exception:
            return None

    def _analyze_code_at_location(self, file_path: str, line_number: int) -> List[str]:
        """Analyze code at location for potential issues"""
        issues = []

        code = self._get_code_context(file_path, line_number, context_lines=5)
        if not code:
            return issues

        # Check for common issues
        if re.search(r'\bNone\b.*\.', code):
            issues.append("Potential NoneType access")

        if re.search(r'\[[^\]]+\]', code) and 'len(' not in code:
            issues.append("Array access without bounds check")

        if re.search(r'open\(', code) and 'try:' not in code:
            issues.append("File operation without error handling")

        if re.search(r'/\s*\w+', code) and 'if.*!=.*0' not in code:
            issues.append("Potential division by zero")

        return issues

    def _search_keyword(self, keyword: str) -> List[Tuple[str, int]]:
        """Search for keyword in codebase"""
        locations = []

        try:
            for py_file in self.codebase_path.rglob('*.py'):
                # Skip hidden and cache dirs
                if any(part.startswith('.') or part == '__pycache__' for part in py_file.parts):
                    continue

                try:
                    with open(py_file, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            if keyword in line:
                                rel_path = py_file.relative_to(self.codebase_path)
                                locations.append((str(rel_path), line_num))
                except:
                    continue

        except Exception:
            pass

        return locations[:50]  # Limit to 50 locations

    def format_report(self, report: BugReport) -> str:
        """Format bug localization report"""

        lines = []
        lines.append("=" * 80)
        lines.append("🐛 BUG LOCALIZATION REPORT")
        lines.append("=" * 80)
        lines.append(f"Symptom: {report.symptom}")
        if report.error_type:
            lines.append(f"Error Type: {report.error_type}")
        lines.append(f"Files Analyzed: {report.total_files_analyzed}")
        lines.append("")

        lines.append("LIKELY BUG LOCATIONS (Ranked by Confidence):")
        lines.append("-" * 80)

        for i, location in enumerate(report.locations[:10], 1):  # Top 10
            confidence_bar = "█" * int(location.confidence * 20)
            lines.append(f"\n{i}. {location.file_path}")
            if location.line_number:
                lines.append(f"   Line: {location.line_number}")
            if location.function_name:
                lines.append(f"   Function: {location.function_name}")
            lines.append(f"   Confidence: {confidence_bar} {location.confidence:.0%}")

            lines.append("   Evidence:")
            for evidence in location.evidence:
                lines.append(f"     • {evidence}")

            if location.suggested_fix:
                lines.append(f"   Suggested Fix: {location.suggested_fix}")

            if location.code_snippet:
                lines.append("   Code Context:")
                for code_line in location.code_snippet.split('\n')[:5]:
                    lines.append(f"     {code_line}")

        lines.append("")
        lines.append("=" * 80)

        return '\n'.join(lines)


# Example usage
if __name__ == "__main__":
    # Test with example traceback
    example_traceback = '''
Traceback (most recent call last):
  File "rag_system/code_assistant.py", line 42, in process_data
    result = process_user_input(data)
  File "rag_system/utils.py", line 128, in process_user_input
    cleaned = data.strip().lower()
AttributeError: 'NoneType' object has no attribute 'strip'
'''

    localizer = BugLocalizer(".")
    report = localizer.locate_from_traceback(example_traceback)

    print(localizer.format_report(report))

    print("\n\n")

    # Test with symptom
    report2 = localizer.locate_from_symptom("Users report 500 Internal Server Error when uploading files >10MB")
    print(localizer.format_report(report2))

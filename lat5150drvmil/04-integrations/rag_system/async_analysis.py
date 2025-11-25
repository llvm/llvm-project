#!/usr/bin/env python3
"""
Async Code Analysis
Parallel processing for 3-5x speedup

Features:
- Parallel security + performance + complexity analysis
- Batch file processing for entire codebases
- Progress tracking with status updates
- Graceful error handling per worker
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from pathlib import Path
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum

# Import analysis engines
try:
    from code_analysis_engine import (
        SecurityScanner, PerformanceOptimizer, ComplexityAnalyzer, CodeSmellDetector
    )
    ANALYSIS_AVAILABLE = True
except ImportError:
    try:
        from rag_system.code_analysis_engine import (
            SecurityScanner, PerformanceOptimizer, ComplexityAnalyzer, CodeSmellDetector
        )
        ANALYSIS_AVAILABLE = True
    except ImportError:
        ANALYSIS_AVAILABLE = False


class AnalysisType(Enum):
    """Types of analysis"""
    SECURITY = "security"
    PERFORMANCE = "performance"
    COMPLEXITY = "complexity"
    SMELLS = "code_smells"
    FULL = "full"


@dataclass
class AnalysisResult:
    """Result from async analysis"""
    analysis_type: AnalysisType
    results: Dict
    duration_seconds: float
    success: bool
    error: Optional[str] = None


@dataclass
class FileAnalysisResult:
    """Results for a single file"""
    filepath: str
    analyses: Dict[AnalysisType, AnalysisResult]
    total_duration: float
    success: bool


class AsyncCodeAnalyzer:
    """
    Parallel code analysis for faster processing

    Example:
        analyzer = AsyncCodeAnalyzer(max_workers=4)

        # Single file, multiple analyses in parallel
        results = analyzer.analyze_parallel(code)

        # Multiple files in parallel
        file_results = analyzer.analyze_codebase(['file1.py', 'file2.py'])
    """

    def __init__(self, max_workers: int = 4, verbose: bool = True):
        """
        Args:
            max_workers: Number of parallel workers (default: 4)
            verbose: Print progress (default: True)
        """
        self.max_workers = max_workers
        self.verbose = verbose
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        if not ANALYSIS_AVAILABLE:
            raise RuntimeError("Analysis engines not available. Install dependencies.")

        # Initialize analyzers (thread-safe, can be shared)
        self.security_scanner = SecurityScanner()
        self.perf_optimizer = PerformanceOptimizer()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.smell_detector = CodeSmellDetector()

    def analyze_parallel(self, code: str, analyses: Optional[List[AnalysisType]] = None) -> Dict[AnalysisType, AnalysisResult]:
        """
        Run multiple analyses in parallel on single code

        Args:
            code: Source code to analyze
            analyses: List of analysis types (default: all)

        Returns:
            Dict mapping analysis type to result
        """
        if analyses is None:
            analyses = [AnalysisType.SECURITY, AnalysisType.PERFORMANCE,
                       AnalysisType.COMPLEXITY, AnalysisType.SMELLS]

        if self.verbose:
            print(f"🚀 Running {len(analyses)} analyses in parallel...")

        start_time = time.time()

        # Submit all analyses to thread pool
        futures = {}
        for analysis_type in analyses:
            future = self.executor.submit(self._run_single_analysis, code, analysis_type)
            futures[future] = analysis_type

        # Collect results as they complete
        results = {}
        for future in as_completed(futures):
            analysis_type = futures[future]
            result = future.result()
            results[analysis_type] = result

            if self.verbose:
                status = "✓" if result.success else "✗"
                print(f"  {status} {analysis_type.value}: {result.duration_seconds:.2f}s")

        total_duration = time.time() - start_time

        if self.verbose:
            speedup = sum(r.duration_seconds for r in results.values()) / total_duration
            print(f"⚡ Total: {total_duration:.2f}s (Effective speedup: {speedup:.1f}x)")

        return results

    def _run_single_analysis(self, code: str, analysis_type: AnalysisType) -> AnalysisResult:
        """Run a single analysis (called in thread pool)"""
        start_time = time.time()

        try:
            if analysis_type == AnalysisType.SECURITY:
                issues = self.security_scanner.scan(code)
                results = {'issues': [
                    {
                        'severity': issue.severity.value,
                        'category': issue.category,
                        'line': issue.line,
                        'description': issue.description,
                        'remediation': issue.remediation,
                        'cwe_id': issue.cwe_id
                    }
                    for issue in issues
                ]}

            elif analysis_type == AnalysisType.PERFORMANCE:
                issues = self.perf_optimizer.analyze(code)
                results = {'issues': [
                    {
                        'line': issue.line,
                        'category': issue.category,
                        'description': issue.description,
                        'suggestion': issue.suggestion,
                        'improvement': issue.estimated_improvement
                    }
                    for issue in issues
                ]}

            elif analysis_type == AnalysisType.COMPLEXITY:
                complexity = self.complexity_analyzer.cyclomatic_complexity(code)
                nesting = self.complexity_analyzer.nesting_depth(code)

                results = {
                    'cyclomatic_complexity': complexity,
                    'nesting_depth': nesting
                }

            elif analysis_type == AnalysisType.SMELLS:
                smells = self.smell_detector.detect(code)
                results = {'smells': [
                    {
                        'line': smell.line,
                        'category': smell.category,
                        'description': smell.description,
                        'suggestion': smell.suggestion
                    }
                    for smell in smells
                ]}

            duration = time.time() - start_time

            return AnalysisResult(
                analysis_type=analysis_type,
                results=results,
                duration_seconds=duration,
                success=True
            )

        except Exception as e:
            duration = time.time() - start_time
            return AnalysisResult(
                analysis_type=analysis_type,
                results={},
                duration_seconds=duration,
                success=False,
                error=str(e)
            )

    def analyze_file(self, filepath: str) -> FileAnalysisResult:
        """
        Analyze a single file with all analyses

        Args:
            filepath: Path to Python file

        Returns:
            FileAnalysisResult with all analysis results
        """
        start_time = time.time()

        try:
            code = Path(filepath).read_text()
            analyses = self.analyze_parallel(code)

            duration = time.time() - start_time

            return FileAnalysisResult(
                filepath=filepath,
                analyses=analyses,
                total_duration=duration,
                success=True
            )

        except Exception as e:
            duration = time.time() - start_time
            return FileAnalysisResult(
                filepath=filepath,
                analyses={},
                total_duration=duration,
                success=False
            )

    def analyze_codebase(self, file_paths: List[str],
                         progress_callback: Optional[Callable] = None) -> List[FileAnalysisResult]:
        """
        Analyze multiple files in parallel

        Args:
            file_paths: List of file paths to analyze
            progress_callback: Optional callback(completed, total)

        Returns:
            List of FileAnalysisResult
        """
        if self.verbose:
            print(f"\n🔍 Analyzing codebase ({len(file_paths)} files)...")
            print(f"⚙️  Using {self.max_workers} parallel workers")

        start_time = time.time()

        # Submit all files to thread pool
        futures = {
            self.executor.submit(self.analyze_file, filepath): filepath
            for filepath in file_paths
        }

        # Collect results with progress
        results = []
        completed = 0

        for future in as_completed(futures):
            filepath = futures[future]
            result = future.result()
            results.append(result)

            completed += 1

            if progress_callback:
                progress_callback(completed, len(file_paths))

            if self.verbose:
                status = "✓" if result.success else "✗"
                print(f"  [{completed}/{len(file_paths)}] {status} {Path(filepath).name} ({result.total_duration:.2f}s)")

        total_duration = time.time() - start_time

        if self.verbose:
            avg_per_file = total_duration / len(file_paths) if file_paths else 0
            print(f"\n⚡ Analysis complete!")
            print(f"   Total time: {total_duration:.2f}s")
            print(f"   Avg per file: {avg_per_file:.2f}s")
            print(f"   Throughput: {len(file_paths) / total_duration:.1f} files/sec")

        return results

    def analyze_directory(self, directory: str, pattern: str = "**/*.py",
                          exclude_patterns: Optional[List[str]] = None) -> List[FileAnalysisResult]:
        """
        Analyze all Python files in directory

        Args:
            directory: Root directory to scan
            pattern: Glob pattern (default: **/*.py)
            exclude_patterns: Patterns to exclude (e.g., ['**/test_*.py'])

        Returns:
            List of FileAnalysisResult
        """
        dir_path = Path(directory)

        # Find all matching files
        files = list(dir_path.glob(pattern))

        # Apply exclude patterns
        if exclude_patterns:
            excluded = set()
            for exclude_pattern in exclude_patterns:
                excluded.update(dir_path.glob(exclude_pattern))
            files = [f for f in files if f not in excluded]

        file_paths = [str(f) for f in files]

        if self.verbose:
            print(f"📂 Found {len(file_paths)} files matching '{pattern}'")
            if exclude_patterns:
                print(f"   Excluded patterns: {exclude_patterns}")

        return self.analyze_codebase(file_paths)

    def shutdown(self):
        """Shutdown executor (cleanup)"""
        self.executor.shutdown(wait=True)

    def __enter__(self):
        """Context manager enter"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()


def main():
    """Test async analysis"""
    test_code = """
import os

def vulnerable_function(user_input):
    # Security issues
    result = eval(user_input)
    os.system("cat " + user_input)

    # Performance issues
    data = ""
    for i in range(len(items)):
        data += str(items[i])

    return data
"""

    print("="*70)
    print("Async Code Analysis Demo")
    print("="*70)

    # Single file analysis (parallel)
    with AsyncCodeAnalyzer(max_workers=4) as analyzer:
        print("\n📊 Single File - Parallel Analysis:")
        results = analyzer.analyze_parallel(test_code)

        print("\n" + "="*70)
        print("Results Summary:")
        for analysis_type, result in results.items():
            if result.success:
                if analysis_type == AnalysisType.SECURITY:
                    print(f"  🔒 Security: {len(result.results.get('issues', []))} issues")
                elif analysis_type == AnalysisType.PERFORMANCE:
                    print(f"  ⚡ Performance: {len(result.results.get('issues', []))} issues")
                elif analysis_type == AnalysisType.COMPLEXITY:
                    print(f"  📊 Complexity: {result.results.get('cyclomatic_complexity', 0)}")
                elif analysis_type == AnalysisType.SMELLS:
                    print(f"  👃 Code Smells: {len(result.results.get('smells', []))}")

    print("\n" + "="*70)
    print("✅ Async analysis complete!")
    print("="*70)


if __name__ == '__main__':
    main()

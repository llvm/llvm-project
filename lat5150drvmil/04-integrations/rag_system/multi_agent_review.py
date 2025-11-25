"""
Multi-Agent Code Review System (Phase 2.1)

Coordinates multiple specialized agents to review code from different perspectives.
Each agent is an expert in a specific domain and provides targeted feedback.

Architecture:
- SecurityReviewAgent: Security vulnerabilities and best practices
- PerformanceReviewAgent: Performance bottlenecks and optimizations
- MaintainabilityReviewAgent: Code maintainability and design patterns
- TestCoverageAgent: Test coverage and test quality
- DocumentationAgent: Documentation completeness and clarity

Features:
- Parallel review execution (5 agents run concurrently)
- Consensus-based recommendations
- Severity-weighted priority ranking
- Interactive discussion mode
"""

import ast
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class ReviewSeverity(Enum):
    """Severity levels for review findings"""
    CRITICAL = 5    # Must fix before merge
    HIGH = 4        # Should fix before merge
    MEDIUM = 3      # Should fix soon
    LOW = 2         # Nice to have
    INFO = 1        # Informational only


@dataclass
class ReviewFinding:
    """A single finding from a review agent"""
    agent: str
    severity: ReviewSeverity
    category: str
    title: str
    description: str
    location: str  # file:line or function name
    suggestion: str
    code_snippet: Optional[str] = None
    confidence: float = 1.0  # 0.0-1.0


@dataclass
class AgentReview:
    """Complete review from a single agent"""
    agent_name: str
    findings: List[ReviewFinding]
    overall_score: float  # 0-10
    summary: str
    execution_time: float


@dataclass
class ConsolidatedReview:
    """Consolidated review from all agents"""
    agent_reviews: Dict[str, AgentReview]
    critical_findings: List[ReviewFinding]
    high_priority_findings: List[ReviewFinding]
    medium_priority_findings: List[ReviewFinding]
    low_priority_findings: List[ReviewFinding]
    overall_score: float
    recommendation: str  # "APPROVE", "APPROVE_WITH_CHANGES", "REQUEST_CHANGES", "REJECT"
    consensus_level: float  # 0.0-1.0 (how much agents agree)


class SecurityReviewAgent:
    """Expert agent for security analysis"""

    def __init__(self):
        self.name = "SecurityExpert"
        # Common security anti-patterns
        self.patterns = {
            'sql_injection': r'(execute|cursor\.execute)\s*\(\s*["\'].*%s.*["\']',
            'hardcoded_secrets': r'(password|api_key|secret|token)\s*=\s*["\'][^"\']+["\']',
            'eval_usage': r'\beval\s*\(',
            'pickle_unsafe': r'\bpickle\.loads\s*\(',
            'shell_injection': r'(subprocess\.|os\.system)\s*\([^)]*\+',
            'weak_crypto': r'(md5|sha1)\s*\(',
            'unsafe_yaml': r'yaml\.load\s*\([^,)]+\)',
        }

    def review(self, code: str, filename: str = "unknown") -> AgentReview:
        """Perform security review"""
        import time
        start_time = time.time()

        findings = []

        # Pattern-based detection
        for vuln_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                line_no = code[:match.start()].count('\n') + 1
                findings.append(ReviewFinding(
                    agent=self.name,
                    severity=ReviewSeverity.HIGH if vuln_type in ['sql_injection', 'shell_injection'] else ReviewSeverity.MEDIUM,
                    category="Security",
                    title=self._get_vuln_title(vuln_type),
                    description=self._get_vuln_description(vuln_type),
                    location=f"{filename}:{line_no}",
                    suggestion=self._get_vuln_fix(vuln_type),
                    code_snippet=match.group(0),
                    confidence=0.85
                ))

        # AST-based analysis
        try:
            tree = ast.parse(code)
            findings.extend(self._ast_security_analysis(tree, filename))
        except SyntaxError:
            pass

        # Calculate score (10 = perfect, 0 = critical issues)
        critical_count = sum(1 for f in findings if f.severity == ReviewSeverity.CRITICAL)
        high_count = sum(1 for f in findings if f.severity == ReviewSeverity.HIGH)
        medium_count = sum(1 for f in findings if f.severity == ReviewSeverity.MEDIUM)

        score = 10.0 - (critical_count * 3.0) - (high_count * 1.5) - (medium_count * 0.5)
        score = max(0.0, min(10.0, score))

        summary = self._generate_summary(findings, score)

        return AgentReview(
            agent_name=self.name,
            findings=findings,
            overall_score=score,
            summary=summary,
            execution_time=time.time() - start_time
        )

    def _ast_security_analysis(self, tree: ast.AST, filename: str) -> List[ReviewFinding]:
        """AST-based security analysis"""
        findings = []

        for node in ast.walk(tree):
            # Check for bare except clauses (can hide security issues)
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    findings.append(ReviewFinding(
                        agent=self.name,
                        severity=ReviewSeverity.LOW,
                        category="Security",
                        title="Bare except clause",
                        description="Catching all exceptions can hide security issues",
                        location=f"{filename}:{node.lineno}",
                        suggestion="Catch specific exceptions instead of using bare 'except:'",
                        confidence=0.9
                    ))

            # Check for assert statements (disabled in production)
            if isinstance(node, ast.Assert):
                findings.append(ReviewFinding(
                    agent=self.name,
                    severity=ReviewSeverity.MEDIUM,
                    category="Security",
                    title="Assert used for validation",
                    description="Assert statements are disabled with -O flag in production",
                    location=f"{filename}:{node.lineno}",
                    suggestion="Use explicit if statements for validation, not assert",
                    confidence=0.85
                ))

        return findings

    def _get_vuln_title(self, vuln_type: str) -> str:
        titles = {
            'sql_injection': "Potential SQL Injection",
            'hardcoded_secrets': "Hardcoded Credentials",
            'eval_usage': "Unsafe eval() Usage",
            'pickle_unsafe': "Unsafe Pickle Deserialization",
            'shell_injection': "Potential Shell Injection",
            'weak_crypto': "Weak Cryptographic Hash",
            'unsafe_yaml': "Unsafe YAML Deserialization",
        }
        return titles.get(vuln_type, "Security Issue")

    def _get_vuln_description(self, vuln_type: str) -> str:
        descriptions = {
            'sql_injection': "String formatting in SQL queries allows injection attacks",
            'hardcoded_secrets': "Credentials should not be hardcoded in source code",
            'eval_usage': "eval() can execute arbitrary code and is a security risk",
            'pickle_unsafe': "pickle.loads() on untrusted data can execute arbitrary code",
            'shell_injection': "String concatenation in shell commands allows injection",
            'weak_crypto': "MD5/SHA1 are cryptographically broken and should not be used",
            'unsafe_yaml': "yaml.load() is unsafe, use yaml.safe_load() instead",
        }
        return descriptions.get(vuln_type, "Security vulnerability detected")

    def _get_vuln_fix(self, vuln_type: str) -> str:
        fixes = {
            'sql_injection': "Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
            'hardcoded_secrets': "Use environment variables or a secrets manager",
            'eval_usage': "Use ast.literal_eval() for safe evaluation or avoid eval entirely",
            'pickle_unsafe': "Use json or validate data source before unpickling",
            'shell_injection': "Use subprocess with list arguments: subprocess.run(['ls', user_input])",
            'weak_crypto': "Use SHA-256 or SHA-3 for cryptographic purposes",
            'unsafe_yaml': "Replace yaml.load() with yaml.safe_load()",
        }
        return fixes.get(vuln_type, "Review and fix security issue")

    def _generate_summary(self, findings: List[ReviewFinding], score: float) -> str:
        if score >= 9.0:
            return f"Excellent security posture. Found {len(findings)} minor issues."
        elif score >= 7.0:
            return f"Good security with {len(findings)} issues to address."
        elif score >= 5.0:
            return f"Moderate security concerns. {len(findings)} issues found."
        else:
            return f"Significant security issues detected ({len(findings)} findings). Review required."


class PerformanceReviewAgent:
    """Expert agent for performance analysis"""

    def __init__(self):
        self.name = "PerformanceExpert"

    def review(self, code: str, filename: str = "unknown") -> AgentReview:
        """Perform performance review"""
        import time
        start_time = time.time()

        findings = []

        # Pattern-based detection
        patterns = {
            'nested_loops': (r'for\s+\w+\s+in.*:\s*\n\s+for\s+\w+\s+in', ReviewSeverity.MEDIUM,
                           "Nested loops detected", "Consider using list comprehensions or numpy operations"),
            'string_concat_loop': (r'for\s+.*:\s*\n\s+.*\+=\s*["\']', ReviewSeverity.MEDIUM,
                                 "String concatenation in loop", "Use ''.join() instead of += for strings"),
            'global_in_loop': (r'for\s+.*:\s*\n\s+global\s+', ReviewSeverity.LOW,
                             "Global variable access in loop", "Cache global lookups outside loop"),
        }

        for pattern_name, (pattern, severity, title, suggestion) in patterns.items():
            matches = re.finditer(pattern, code, re.MULTILINE)
            for match in matches:
                line_no = code[:match.start()].count('\n') + 1
                findings.append(ReviewFinding(
                    agent=self.name,
                    severity=severity,
                    category="Performance",
                    title=title,
                    description=f"Performance anti-pattern detected: {pattern_name}",
                    location=f"{filename}:{line_no}",
                    suggestion=suggestion,
                    code_snippet=match.group(0)[:100],
                    confidence=0.75
                ))

        # AST-based analysis
        try:
            tree = ast.parse(code)
            findings.extend(self._ast_performance_analysis(tree, filename))
        except SyntaxError:
            pass

        # Calculate score
        high_count = sum(1 for f in findings if f.severity == ReviewSeverity.HIGH)
        medium_count = sum(1 for f in findings if f.severity == ReviewSeverity.MEDIUM)
        low_count = sum(1 for f in findings if f.severity == ReviewSeverity.LOW)

        score = 10.0 - (high_count * 2.0) - (medium_count * 1.0) - (low_count * 0.3)
        score = max(0.0, min(10.0, score))

        summary = f"Performance score: {score:.1f}/10. Found {len(findings)} optimization opportunities."

        return AgentReview(
            agent_name=self.name,
            findings=findings,
            overall_score=score,
            summary=summary,
            execution_time=time.time() - start_time
        )

    def _ast_performance_analysis(self, tree: ast.AST, filename: str) -> List[ReviewFinding]:
        """AST-based performance analysis"""
        findings = []

        for node in ast.walk(tree):
            # Detect list comprehension that could be generator
            if isinstance(node, ast.ListComp):
                findings.append(ReviewFinding(
                    agent=self.name,
                    severity=ReviewSeverity.LOW,
                    category="Performance",
                    title="Consider using generator expression",
                    description="Generator expressions are more memory efficient for large datasets",
                    location=f"{filename}:{node.lineno}",
                    suggestion="Replace [...] with (...) if items are only iterated once",
                    confidence=0.6
                ))

        return findings


class MaintainabilityReviewAgent:
    """Expert agent for code maintainability"""

    def __init__(self):
        self.name = "MaintainabilityExpert"

    def review(self, code: str, filename: str = "unknown") -> AgentReview:
        """Perform maintainability review"""
        import time
        start_time = time.time()

        findings = []

        # Analyze code complexity
        try:
            tree = ast.parse(code)
            findings.extend(self._analyze_complexity(tree, filename, code))
            findings.extend(self._analyze_naming(tree, filename))
            findings.extend(self._analyze_structure(tree, filename))
        except SyntaxError:
            findings.append(ReviewFinding(
                agent=self.name,
                severity=ReviewSeverity.CRITICAL,
                category="Maintainability",
                title="Syntax Error",
                description="Code contains syntax errors",
                location=filename,
                suggestion="Fix syntax errors before reviewing",
                confidence=1.0
            ))

        # Calculate score
        critical_count = sum(1 for f in findings if f.severity == ReviewSeverity.CRITICAL)
        high_count = sum(1 for f in findings if f.severity == ReviewSeverity.HIGH)
        medium_count = sum(1 for f in findings if f.severity == ReviewSeverity.MEDIUM)

        score = 10.0 - (critical_count * 3.0) - (high_count * 1.5) - (medium_count * 0.5)
        score = max(0.0, min(10.0, score))

        summary = f"Maintainability score: {score:.1f}/10. Code {'is well-structured' if score >= 8 else 'needs refactoring'}."

        return AgentReview(
            agent_name=self.name,
            findings=findings,
            overall_score=score,
            summary=summary,
            execution_time=time.time() - start_time
        )

    def _analyze_complexity(self, tree: ast.AST, filename: str, code: str) -> List[ReviewFinding]:
        """Analyze cyclomatic complexity"""
        findings = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_complexity(node)
                if complexity > 10:
                    severity = ReviewSeverity.HIGH if complexity > 15 else ReviewSeverity.MEDIUM
                    findings.append(ReviewFinding(
                        agent=self.name,
                        severity=severity,
                        category="Maintainability",
                        title=f"High complexity function: {node.name}",
                        description=f"Cyclomatic complexity: {complexity} (threshold: 10)",
                        location=f"{filename}:{node.lineno}",
                        suggestion="Break down into smaller functions or reduce branching",
                        confidence=0.95
                    ))

                # Check function length
                func_lines = len([n for n in ast.walk(node) if hasattr(n, 'lineno')])
                if func_lines > 50:
                    findings.append(ReviewFinding(
                        agent=self.name,
                        severity=ReviewSeverity.MEDIUM,
                        category="Maintainability",
                        title=f"Long function: {node.name}",
                        description=f"Function has {func_lines} lines (recommended: <50)",
                        location=f"{filename}:{node.lineno}",
                        suggestion="Extract logical blocks into separate functions",
                        confidence=0.9
                    ))

        return findings

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

    def _analyze_naming(self, tree: ast.AST, filename: str) -> List[ReviewFinding]:
        """Analyze naming conventions"""
        findings = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for non-snake_case function names
                if not re.match(r'^[a-z_][a-z0-9_]*$', node.name) and not node.name.startswith('__'):
                    findings.append(ReviewFinding(
                        agent=self.name,
                        severity=ReviewSeverity.LOW,
                        category="Maintainability",
                        title=f"Non-standard function name: {node.name}",
                        description="Function names should use snake_case",
                        location=f"{filename}:{node.lineno}",
                        suggestion=f"Rename to {self._to_snake_case(node.name)}",
                        confidence=0.9
                    ))

                # Check for single-letter names (except common ones)
                if len(node.name) == 1 and node.name not in ['x', 'y', 'z', 'i', 'j', 'k']:
                    findings.append(ReviewFinding(
                        agent=self.name,
                        severity=ReviewSeverity.LOW,
                        category="Maintainability",
                        title=f"Single-letter function name: {node.name}",
                        description="Use descriptive function names",
                        location=f"{filename}:{node.lineno}",
                        suggestion="Choose a more descriptive name",
                        confidence=0.8
                    ))

        return findings

    def _to_snake_case(self, name: str) -> str:
        """Convert to snake_case"""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def _analyze_structure(self, tree: ast.AST, filename: str) -> List[ReviewFinding]:
        """Analyze code structure"""
        findings = []

        # Check for too many parameters
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                param_count = len(node.args.args)
                if param_count > 5:
                    findings.append(ReviewFinding(
                        agent=self.name,
                        severity=ReviewSeverity.MEDIUM,
                        category="Maintainability",
                        title=f"Too many parameters: {node.name}",
                        description=f"Function has {param_count} parameters (recommended: ≤5)",
                        location=f"{filename}:{node.lineno}",
                        suggestion="Consider using a parameter object or reducing parameters",
                        confidence=0.85
                    ))

        return findings


class TestCoverageAgent:
    """Expert agent for test coverage analysis"""

    def __init__(self):
        self.name = "TestCoverageExpert"

    def review(self, code: str, filename: str = "unknown") -> AgentReview:
        """Perform test coverage review"""
        import time
        start_time = time.time()

        findings = []

        # Check if this is a test file
        is_test_file = 'test_' in filename or filename.endswith('_test.py')

        if is_test_file:
            findings.extend(self._analyze_test_quality(code, filename))
        else:
            findings.extend(self._analyze_testability(code, filename))

        # Calculate score
        high_count = sum(1 for f in findings if f.severity == ReviewSeverity.HIGH)
        medium_count = sum(1 for f in findings if f.severity == ReviewSeverity.MEDIUM)

        score = 10.0 - (high_count * 2.0) - (medium_count * 0.75)
        score = max(0.0, min(10.0, score))

        summary = f"Test quality score: {score:.1f}/10. {'Good test coverage' if score >= 8 else 'More tests recommended'}."

        return AgentReview(
            agent_name=self.name,
            findings=findings,
            overall_score=score,
            summary=summary,
            execution_time=time.time() - start_time
        )

    def _analyze_test_quality(self, code: str, filename: str) -> List[ReviewFinding]:
        """Analyze test file quality"""
        findings = []

        # Check for test count
        test_count = len(re.findall(r'def test_\w+', code))
        if test_count == 0:
            findings.append(ReviewFinding(
                agent=self.name,
                severity=ReviewSeverity.HIGH,
                category="Testing",
                title="No tests found",
                description="Test file contains no test functions",
                location=filename,
                suggestion="Add test functions (def test_...)",
                confidence=1.0
            ))

        # Check for assertions
        assertion_count = len(re.findall(r'\bassert\b', code))
        if assertion_count == 0 and test_count > 0:
            findings.append(ReviewFinding(
                agent=self.name,
                severity=ReviewSeverity.HIGH,
                category="Testing",
                title="No assertions in tests",
                description="Tests should contain assertions to verify behavior",
                location=filename,
                suggestion="Add assert statements to verify expected outcomes",
                confidence=0.95
            ))

        return findings

    def _analyze_testability(self, code: str, filename: str) -> List[ReviewFinding]:
        """Analyze code testability"""
        findings = []

        try:
            tree = ast.parse(code)

            # Count functions that might need tests
            function_count = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and not n.name.startswith('_')])

            if function_count > 0:
                findings.append(ReviewFinding(
                    agent=self.name,
                    severity=ReviewSeverity.MEDIUM,
                    category="Testing",
                    title="Consider adding tests",
                    description=f"Found {function_count} public functions without visible test coverage",
                    location=filename,
                    suggestion=f"Create {filename.replace('.py', '_test.py')} with test cases",
                    confidence=0.7
                ))
        except SyntaxError:
            pass

        return findings


class DocumentationAgent:
    """Expert agent for documentation analysis"""

    def __init__(self):
        self.name = "DocumentationExpert"

    def review(self, code: str, filename: str = "unknown") -> AgentReview:
        """Perform documentation review"""
        import time
        start_time = time.time()

        findings = []

        try:
            tree = ast.parse(code)
            findings.extend(self._analyze_docstrings(tree, filename))
            findings.extend(self._analyze_comments(code, filename))
        except SyntaxError:
            pass

        # Calculate score
        high_count = sum(1 for f in findings if f.severity == ReviewSeverity.HIGH)
        medium_count = sum(1 for f in findings if f.severity == ReviewSeverity.MEDIUM)

        score = 10.0 - (high_count * 1.5) - (medium_count * 0.5)
        score = max(0.0, min(10.0, score))

        summary = f"Documentation score: {score:.1f}/10. {'Well documented' if score >= 8 else 'Needs more documentation'}."

        return AgentReview(
            agent_name=self.name,
            findings=findings,
            overall_score=score,
            summary=summary,
            execution_time=time.time() - start_time
        )

    def _analyze_docstrings(self, tree: ast.AST, filename: str) -> List[ReviewFinding]:
        """Analyze docstring coverage"""
        findings = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node)

                if docstring is None:
                    # Public functions/classes should have docstrings
                    if not node.name.startswith('_'):
                        findings.append(ReviewFinding(
                            agent=self.name,
                            severity=ReviewSeverity.MEDIUM,
                            category="Documentation",
                            title=f"Missing docstring: {node.name}",
                            description=f"{'Function' if isinstance(node, ast.FunctionDef) else 'Class'} lacks documentation",
                            location=f"{filename}:{node.lineno}",
                            suggestion="Add docstring describing purpose, parameters, and return value",
                            confidence=0.9
                        ))
                elif len(docstring) < 20:
                    findings.append(ReviewFinding(
                        agent=self.name,
                        severity=ReviewSeverity.LOW,
                        category="Documentation",
                        title=f"Brief docstring: {node.name}",
                        description="Docstring is very short and may lack detail",
                        location=f"{filename}:{node.lineno}",
                        suggestion="Expand docstring with more details",
                        confidence=0.7
                    ))

        return findings

    def _analyze_comments(self, code: str, filename: str) -> List[ReviewFinding]:
        """Analyze code comments"""
        findings = []

        # Count lines of code vs comments
        lines = code.split('\n')
        code_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
        comment_lines = len([l for l in lines if l.strip().startswith('#')])

        if code_lines > 50 and comment_lines == 0:
            findings.append(ReviewFinding(
                agent=self.name,
                severity=ReviewSeverity.LOW,
                category="Documentation",
                title="No inline comments",
                description=f"{code_lines} lines of code with no comments",
                location=filename,
                suggestion="Add comments explaining complex logic",
                confidence=0.6
            ))

        return findings


class MultiAgentReviewer:
    """Coordinates multiple review agents"""

    def __init__(self, max_workers: int = 5):
        self.agents = {
            'security': SecurityReviewAgent(),
            'performance': PerformanceReviewAgent(),
            'maintainability': MaintainabilityReviewAgent(),
            'testing': TestCoverageAgent(),
            'documentation': DocumentationAgent()
        }
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def review_code(self, code: str, filename: str = "unknown") -> ConsolidatedReview:
        """Run all agents in parallel and consolidate results"""

        # Submit all reviews in parallel
        futures = {
            name: self.executor.submit(agent.review, code, filename)
            for name, agent in self.agents.items()
        }

        # Collect results
        agent_reviews = {}
        for name, future in futures.items():
            try:
                agent_reviews[name] = future.result()
            except Exception as e:
                # Create error review if agent fails
                agent_reviews[name] = AgentReview(
                    agent_name=name,
                    findings=[],
                    overall_score=5.0,
                    summary=f"Review failed: {str(e)}",
                    execution_time=0.0
                )

        # Consolidate findings by severity
        all_findings = []
        for review in agent_reviews.values():
            all_findings.extend(review.findings)

        critical = [f for f in all_findings if f.severity == ReviewSeverity.CRITICAL]
        high = [f for f in all_findings if f.severity == ReviewSeverity.HIGH]
        medium = [f for f in all_findings if f.severity == ReviewSeverity.MEDIUM]
        low = [f for f in all_findings if f.severity == ReviewSeverity.LOW]

        # Calculate overall score (weighted average)
        weights = {
            'security': 0.3,
            'performance': 0.2,
            'maintainability': 0.25,
            'testing': 0.15,
            'documentation': 0.1
        }

        overall_score = sum(
            agent_reviews[name].overall_score * weight
            for name, weight in weights.items()
            if name in agent_reviews
        )

        # Calculate consensus level (how much agents agree)
        scores = [r.overall_score for r in agent_reviews.values()]
        avg_score = sum(scores) / len(scores)
        variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        consensus = 1.0 - min(variance / 10.0, 1.0)  # Normalize to 0-1

        # Determine recommendation
        if len(critical) > 0:
            recommendation = "REJECT"
        elif len(high) > 5 or overall_score < 5.0:
            recommendation = "REQUEST_CHANGES"
        elif len(high) > 0 or len(medium) > 3:
            recommendation = "APPROVE_WITH_CHANGES"
        else:
            recommendation = "APPROVE"

        return ConsolidatedReview(
            agent_reviews=agent_reviews,
            critical_findings=critical,
            high_priority_findings=high,
            medium_priority_findings=medium,
            low_priority_findings=low,
            overall_score=overall_score,
            recommendation=recommendation,
            consensus_level=consensus
        )

    def format_review_report(self, review: ConsolidatedReview) -> str:
        """Format consolidated review as readable report"""

        lines = []
        lines.append("=" * 80)
        lines.append("MULTI-AGENT CODE REVIEW REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Overall summary
        lines.append(f"Overall Score: {review.overall_score:.1f}/10")
        lines.append(f"Recommendation: {review.recommendation}")
        lines.append(f"Agent Consensus: {review.consensus_level:.1%}")
        lines.append("")

        # Agent summaries
        lines.append("AGENT REVIEWS:")
        lines.append("-" * 80)
        for name, agent_review in review.agent_reviews.items():
            lines.append(f"  {name.upper()}: {agent_review.overall_score:.1f}/10 ({len(agent_review.findings)} findings)")
            lines.append(f"    {agent_review.summary}")
            lines.append("")

        # Critical findings
        if review.critical_findings:
            lines.append("CRITICAL ISSUES (Must Fix):")
            lines.append("-" * 80)
            for finding in review.critical_findings:
                lines.append(f"  [{finding.agent}] {finding.title}")
                lines.append(f"    Location: {finding.location}")
                lines.append(f"    {finding.description}")
                lines.append(f"    Fix: {finding.suggestion}")
                lines.append("")

        # High priority findings
        if review.high_priority_findings:
            lines.append("HIGH PRIORITY ISSUES (Should Fix):")
            lines.append("-" * 80)
            for finding in review.high_priority_findings[:5]:  # Show top 5
                lines.append(f"  [{finding.agent}] {finding.title}")
                lines.append(f"    Location: {finding.location}")
                lines.append(f"    Fix: {finding.suggestion}")
                lines.append("")

            if len(review.high_priority_findings) > 5:
                lines.append(f"  ... and {len(review.high_priority_findings) - 5} more high priority issues")
                lines.append("")

        # Summary counts
        lines.append("SUMMARY:")
        lines.append("-" * 80)
        lines.append(f"  Critical: {len(review.critical_findings)}")
        lines.append(f"  High:     {len(review.high_priority_findings)}")
        lines.append(f"  Medium:   {len(review.medium_priority_findings)}")
        lines.append(f"  Low:      {len(review.low_priority_findings)}")
        lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    # Test code with various issues
    test_code = '''
import os
import pickle

def processUserData(username, password, email, phone, address):
    """Process user data"""
    # SQL Injection vulnerability
    query = "SELECT * FROM users WHERE username = '%s'" % username
    cursor.execute(query)

    # Hardcoded credential
    api_key = "sk-1234567890abcdef"

    # Unsafe deserialization
    data = pickle.loads(user_input)

    # Complex nested loops
    for i in range(100):
        for j in range(100):
            for k in range(100):
                result += i * j * k

    return result
'''

    reviewer = MultiAgentReviewer()
    consolidated = reviewer.review_code(test_code, "example.py")

    print(reviewer.format_review_report(consolidated))

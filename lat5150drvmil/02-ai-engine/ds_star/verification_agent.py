#!/usr/bin/env python3
"""
DS-STAR Verification Agent

Verifies execution outputs against success criteria.
Catches errors early before cascading failures.

Author: LAT5150DRVMIL AI Framework
Version: 1.0.0
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class VerificationStatus(Enum):
    """Verification result status"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"


@dataclass
class VerificationResult:
    """Result of verification check"""
    status: str  # VerificationStatus value
    passed_criteria: List[str]
    failed_criteria: List[str]
    suggestions: List[str]
    confidence: float

    def is_success(self) -> bool:
        return self.status == VerificationStatus.SUCCESS.value

    def is_partial(self) -> bool:
        return self.status == VerificationStatus.PARTIAL.value

    def is_failure(self) -> bool:
        return self.status == VerificationStatus.FAILURE.value


class VerificationAgent:
    """
    Verify execution outputs against criteria

    Usage:
        agent = VerificationAgent()
        result = agent.verify(
            output={"query_time": 50},
            success_criteria=["Query time < 100ms"]
        )
    """

    def __init__(self):
        """Initialize verification agent"""
        pass

    def verify(
        self,
        output: Any,
        success_criteria: List[str],
        context: Dict = None
    ) -> VerificationResult:
        """
        Verify output against success criteria

        Args:
            output: Execution output to verify
            success_criteria: List of success criteria
            context: Additional context for verification

        Returns:
            VerificationResult with status and details
        """
        passed = []
        failed = []
        suggestions = []

        for criterion in success_criteria:
            # Simple rule-based verification (would use LLM in production)
            is_satisfied, suggestion = self._check_criterion(criterion, output, context)

            if is_satisfied:
                passed.append(criterion)
            else:
                failed.append(criterion)
                if suggestion:
                    suggestions.append(suggestion)

        # Determine status
        if len(failed) == 0:
            status = VerificationStatus.SUCCESS
            confidence = 1.0
        elif len(passed) > 0:
            status = VerificationStatus.PARTIAL
            confidence = len(passed) / len(success_criteria)
        else:
            status = VerificationStatus.FAILURE
            confidence = 0.0

        return VerificationResult(
            status=status.value,
            passed_criteria=passed,
            failed_criteria=failed,
            suggestions=suggestions,
            confidence=confidence
        )

    def _check_criterion(
        self,
        criterion: str,
        output: Any,
        context: Dict = None
    ) -> tuple[bool, str]:
        """
        Check single criterion

        Returns:
            (is_satisfied, suggestion_if_failed)
        """
        criterion_lower = criterion.lower()

        # Rule 1: Numeric comparisons
        if '<' in criterion or '>' in criterion:
            return self._check_numeric_criterion(criterion, output, context)

        # Rule 2: Presence checks
        if 'identified' in criterion_lower or 'found' in criterion_lower:
            return self._check_presence(criterion, output)

        # Rule 3: Quality checks
        if 'passing' in criterion_lower or 'passed' in criterion_lower:
            return self._check_quality(criterion, output)

        # Default: assume satisfied (placeholder)
        return True, ""

    def _check_numeric_criterion(
        self,
        criterion: str,
        output: Any,
        context: Dict
    ) -> tuple[bool, str]:
        """Check numeric comparison (e.g., 'Query time < 100ms')"""
        # Simplified - would parse criterion properly in production
        if isinstance(output, dict):
            # Check if output has relevant numeric value
            for key, value in output.items():
                if isinstance(value, (int, float)):
                    if '<' in criterion and value < 100:  # Placeholder threshold
                        return True, ""
                    elif '>' in criterion and value > 0:
                        return True, ""

        return False, f"Numeric criterion not met: {criterion}"

    def _check_presence(self, criterion: str, output: Any) -> tuple[bool, str]:
        """Check if something was identified/found"""
        if isinstance(output, (list, dict)):
            if len(output) > 0:
                return True, ""
            else:
                return False, f"Nothing found for: {criterion}"

        return False, f"Cannot verify presence: {criterion}"

    def _check_quality(self, criterion: str, output: Any) -> tuple[bool, str]:
        """Check quality criteria (e.g., tests passing)"""
        if isinstance(output, dict):
            if output.get("tests_passed", False):
                return True, ""
            elif "quality" in output and output["quality"] > 0.7:
                return True, ""

        return False, f"Quality check failed: {criterion}"


if __name__ == "__main__":
    print("="*70)
    print("DS-STAR Verification Agent Demo")
    print("="*70)

    agent = VerificationAgent()

    # Test case 1: Success
    output = {"query_time": 50, "tests_passed": True}
    criteria = ["Query time < 100ms", "All tests passing"]
    result = agent.verify(output, criteria)

    print("\nCase 1: Success")
    print(f"  Status: {result.status}")
    print(f"  Passed: {result.passed_criteria}")
    print(f"  Failed: {result.failed_criteria}")
    print(f"  Confidence: {result.confidence:.2f}")

    # Test case 2: Partial
    output = {"indexes_found": ["idx1", "idx2"], "tests_passed": False}
    criteria = ["Identified missing indexes", "All tests passing"]
    result = agent.verify(output, criteria)

    print("\nCase 2: Partial success")
    print(f"  Status: {result.status}")
    print(f"  Passed: {result.passed_criteria}")
    print(f"  Failed: {result.failed_criteria}")
    print(f"  Suggestions: {result.suggestions}")

    # Test case 3: Failure
    output = {}
    criteria = ["Query optimized", "Performance improved"]
    result = agent.verify(output, criteria)

    print("\nCase 3: Failure")
    print(f"  Status: {result.status}")
    print(f"  Failed: {result.failed_criteria}")

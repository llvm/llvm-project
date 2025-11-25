#!/usr/bin/env python3
"""
Mixture of Experts Router

Routes queries to the most appropriate expert model(s) based on query analysis.
Implements intelligent routing with confidence scoring and multi-expert selection.
"""

import re
import json
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class ExpertDomain(Enum):
    """Expert domains matching agent categories"""
    CODE = "code"                      # Code generation, debugging, optimization
    DATABASE = "database"              # SQL, data modeling, query optimization
    SECURITY = "security"              # Security analysis, vulnerability detection
    INFRASTRUCTURE = "infrastructure"  # DevOps, deployment, configuration
    DOCUMENTATION = "documentation"    # Writing, explaining, documentation
    ANALYSIS = "analysis"              # Data analysis, performance analysis
    TESTING = "testing"                # QA, test generation, validation
    STRATEGIC = "strategic"            # Architecture, planning, design
    OPERATIONS = "operations"          # Monitoring, maintenance, troubleshooting


@dataclass
class ExpertCandidate:
    """A candidate expert for routing"""
    domain: ExpertDomain
    confidence: float  # 0.0 to 1.0
    reasoning: str
    model_path: Optional[str] = None
    priority: int = 5  # 1-10, higher = more urgent


@dataclass
class RoutingDecision:
    """Final routing decision for a query"""
    primary_expert: ExpertCandidate
    secondary_experts: List[ExpertCandidate] = field(default_factory=list)
    should_ensemble: bool = False
    routing_strategy: str = "single"  # single, parallel, sequential, ensemble
    estimated_complexity: str = "medium"  # simple, medium, hard


class MoERouter:
    """
    Intelligent router that selects the best expert(s) for each query.

    Uses pattern matching, keyword analysis, and heuristics to route queries
    to specialized expert models.
    """

    def __init__(self, enable_multi_expert: bool = True, confidence_threshold: float = 0.3):
        """
        Initialize the MoE router.

        Args:
            enable_multi_expert: Allow routing to multiple experts
            confidence_threshold: Minimum confidence to include secondary experts
        """
        self.enable_multi_expert = enable_multi_expert
        self.confidence_threshold = confidence_threshold

        # Expert domain patterns
        self.domain_patterns = self._init_domain_patterns()

        # Expert capabilities
        self.expert_capabilities = self._init_expert_capabilities()

        # Routing statistics
        self.routing_stats = {domain: 0 for domain in ExpertDomain}

    def _init_domain_patterns(self) -> Dict[ExpertDomain, Dict[str, List[str]]]:
        """Initialize keyword patterns for each expert domain."""
        return {
            ExpertDomain.CODE: {
                "keywords": ["code", "function", "class", "implement", "refactor", "bug", "debug",
                           "compile", "syntax", "algorithm", "programming", "python", "java", "rust"],
                "patterns": [
                    r"\b(write|create|implement|fix|debug|refactor)\s+(a\s+)?(\w+\s+)?(function|class|method|code)",
                    r"\b(optimize|improve|fix)\s+(\w+\s+)?(code|algorithm|implementation)",
                    r"\b(syntax\s+error|compilation\s+error|runtime\s+error)"
                ]
            },
            ExpertDomain.DATABASE: {
                "keywords": ["sql", "query", "database", "table", "schema", "postgres", "mysql",
                           "mongodb", "index", "transaction", "join", "select", "insert", "update"],
                "patterns": [
                    r"\b(sql|query|database|db)\b",
                    r"\b(select|insert|update|delete|create\s+table|alter\s+table)",
                    r"\b(optimize|analyze)\s+(\w+\s+)?(query|queries)",
                    r"\b(schema|data\s+model|er\s+diagram)"
                ]
            },
            ExpertDomain.SECURITY: {
                "keywords": ["security", "vulnerability", "exploit", "penetration", "authentication",
                           "authorization", "encryption", "tpm", "secure", "attack", "threat", "cve"],
                "patterns": [
                    r"\b(security|vulnerability|exploit|penetration|pentest)",
                    r"\b(sql\s+injection|xss|csrf|authentication|authorization)",
                    r"\b(encrypt|decrypt|hash|signature|certificate|tpm)",
                    r"\b(threat|attack|malware|virus)"
                ]
            },
            ExpertDomain.INFRASTRUCTURE: {
                "keywords": ["deploy", "docker", "kubernetes", "container", "infrastructure",
                           "devops", "ci/cd", "pipeline", "terraform", "ansible", "nginx", "server"],
                "patterns": [
                    r"\b(deploy|deployment|infrastructure|devops)",
                    r"\b(docker|kubernetes|k8s|container|containerize)",
                    r"\b(ci/cd|pipeline|jenkins|github\s+actions)",
                    r"\b(terraform|ansible|chef|puppet|configuration\s+management)"
                ]
            },
            ExpertDomain.DOCUMENTATION: {
                "keywords": ["document", "explain", "describe", "write", "readme", "guide",
                           "tutorial", "api\s+docs", "comment", "docstring"],
                "patterns": [
                    r"\b(write|create|generate)\s+(\w+\s+)?(documentation|docs|readme|guide)",
                    r"\b(explain|describe|document)\s+(\w+\s+)?(how|what|why)",
                    r"\b(api\s+docs|docstring|comment|inline\s+documentation)"
                ]
            },
            ExpertDomain.ANALYSIS: {
                "keywords": ["analyze", "analysis", "performance", "profile", "benchmark",
                           "metrics", "statistics", "data\s+analysis", "visualize"],
                "patterns": [
                    r"\b(analyze|analysis|examine|investigate)",
                    r"\b(performance|profiling|benchmark|optimization)",
                    r"\b(metrics|statistics|kpi|dashboard)",
                    r"\b(data\s+analysis|data\s+science|visualization)"
                ]
            },
            ExpertDomain.TESTING: {
                "keywords": ["test", "testing", "qa", "quality", "unit\s+test", "integration\s+test",
                           "validation", "verify", "coverage"],
                "patterns": [
                    r"\b(test|testing|qa|quality\s+assurance)",
                    r"\b(unit\s+test|integration\s+test|e2e\s+test|system\s+test)",
                    r"\b(validation|verify|validate|check)",
                    r"\b(coverage|test\s+coverage)"
                ]
            },
            ExpertDomain.STRATEGIC: {
                "keywords": ["architecture", "design", "plan", "strategy", "roadmap",
                           "system\s+design", "scalability", "high-level"],
                "patterns": [
                    r"\b(architecture|system\s+design|architectural\s+pattern)",
                    r"\b(design|plan|strategy|roadmap)",
                    r"\b(scalability|scale|distributed\s+system)",
                    r"\b(high-level|overview|blueprint)"
                ]
            },
            ExpertDomain.OPERATIONS: {
                "keywords": ["monitor", "maintenance", "troubleshoot", "incident", "alert",
                           "logging", "observability", "ops", "sre"],
                "patterns": [
                    r"\b(monitor|monitoring|observability|telemetry)",
                    r"\b(maintenance|troubleshoot|debug|diagnose)",
                    r"\b(incident|alert|notification|on-call)",
                    r"\b(logging|logs|tracing|metrics)"
                ]
            }
        }

    def _init_expert_capabilities(self) -> Dict[ExpertDomain, List[str]]:
        """Initialize capability descriptions for each expert."""
        return {
            ExpertDomain.CODE: [
                "Code generation and synthesis",
                "Bug fixing and debugging",
                "Code refactoring and optimization",
                "Algorithm implementation",
                "Multi-language support (Python, Java, Rust, C++)"
            ],
            ExpertDomain.DATABASE: [
                "SQL query generation and optimization",
                "Schema design and data modeling",
                "Index optimization",
                "Transaction management",
                "Database migration"
            ],
            ExpertDomain.SECURITY: [
                "Vulnerability detection and analysis",
                "Security code review",
                "Penetration testing guidance",
                "Cryptographic implementation",
                "TPM and hardware security"
            ],
            ExpertDomain.INFRASTRUCTURE: [
                "Containerization and orchestration",
                "CI/CD pipeline design",
                "Infrastructure as Code (Terraform, Ansible)",
                "Cloud deployment strategies",
                "Configuration management"
            ],
            ExpertDomain.DOCUMENTATION: [
                "Technical documentation generation",
                "API documentation",
                "Code comments and docstrings",
                "README and guide creation",
                "Explanation and tutorial writing"
            ],
            ExpertDomain.ANALYSIS: [
                "Performance analysis and profiling",
                "Data analysis and visualization",
                "Benchmarking and comparison",
                "Metrics collection and reporting",
                "Root cause analysis"
            ],
            ExpertDomain.TESTING: [
                "Test case generation",
                "Unit and integration testing",
                "Test automation",
                "Quality assurance",
                "Code coverage analysis"
            ],
            ExpertDomain.STRATEGIC: [
                "System architecture design",
                "Technical planning and roadmaps",
                "Scalability analysis",
                "Technology selection",
                "High-level system design"
            ],
            ExpertDomain.OPERATIONS: [
                "System monitoring and alerting",
                "Troubleshooting and diagnostics",
                "Incident response",
                "Log analysis",
                "SRE practices"
            ]
        }

    def route(self, query: str, context: Optional[Dict] = None) -> RoutingDecision:
        """
        Route a query to the most appropriate expert(s).

        Args:
            query: The user query
            context: Optional context (previous messages, user preferences, etc.)

        Returns:
            RoutingDecision with primary and secondary experts
        """
        # Analyze query to get all candidate experts with confidence scores
        candidates = self._analyze_query(query, context)

        if not candidates:
            # Fallback to CODE expert for general queries
            candidates = [ExpertCandidate(
                domain=ExpertDomain.CODE,
                confidence=0.5,
                reasoning="Default fallback for unclear query"
            )]

        # Sort by confidence
        candidates.sort(key=lambda x: x.confidence, reverse=True)

        # Select primary expert
        primary_expert = candidates[0]

        # Select secondary experts if enabled
        secondary_experts = []
        if self.enable_multi_expert and len(candidates) > 1:
            for candidate in candidates[1:]:
                if candidate.confidence >= self.confidence_threshold:
                    secondary_experts.append(candidate)

        # Determine routing strategy
        routing_strategy = self._determine_routing_strategy(
            primary_expert, secondary_experts, query
        )

        # Should we ensemble?
        should_ensemble = (
            len(secondary_experts) > 0 and
            routing_strategy == "ensemble" and
            primary_expert.confidence < 0.8
        )

        # Estimate complexity
        complexity = self._estimate_complexity(query, context)

        # Update statistics
        self.routing_stats[primary_expert.domain] += 1

        return RoutingDecision(
            primary_expert=primary_expert,
            secondary_experts=secondary_experts,
            should_ensemble=should_ensemble,
            routing_strategy=routing_strategy,
            estimated_complexity=complexity
        )

    def _analyze_query(self, query: str, context: Optional[Dict]) -> List[ExpertCandidate]:
        """Analyze query and return candidate experts with confidence scores."""
        candidates = []
        query_lower = query.lower()

        for domain, patterns in self.domain_patterns.items():
            confidence = 0.0
            reasoning_parts = []

            # Check keywords
            keyword_matches = sum(
                1 for keyword in patterns["keywords"]
                if re.search(r'\b' + keyword + r'\b', query_lower, re.IGNORECASE)
            )
            keyword_confidence = min(keyword_matches * 0.15, 0.6)
            confidence += keyword_confidence

            if keyword_matches > 0:
                reasoning_parts.append(f"{keyword_matches} keyword matches")

            # Check regex patterns
            pattern_matches = sum(
                1 for pattern in patterns["patterns"]
                if re.search(pattern, query_lower, re.IGNORECASE)
            )
            pattern_confidence = min(pattern_matches * 0.25, 0.5)
            confidence += pattern_confidence

            if pattern_matches > 0:
                reasoning_parts.append(f"{pattern_matches} pattern matches")

            # Context bonus
            if context:
                if context.get("previous_domain") == domain:
                    confidence += 0.1
                    reasoning_parts.append("context continuity")

            # Normalize confidence to [0, 1]
            confidence = min(confidence, 1.0)

            if confidence > 0.0:
                candidates.append(ExpertCandidate(
                    domain=domain,
                    confidence=confidence,
                    reasoning="; ".join(reasoning_parts) if reasoning_parts else "No strong match"
                ))

        return candidates

    def _determine_routing_strategy(
        self,
        primary: ExpertCandidate,
        secondaries: List[ExpertCandidate],
        query: str
    ) -> str:
        """Determine the best routing strategy."""
        if not secondaries:
            return "single"

        # High confidence primary -> single expert
        if primary.confidence > 0.8:
            return "single"

        # Multiple strong candidates -> ensemble
        strong_secondaries = [s for s in secondaries if s.confidence > 0.5]
        if len(strong_secondaries) >= 2:
            return "ensemble"

        # One strong secondary -> parallel consultation
        if len(strong_secondaries) == 1:
            return "parallel"

        # Weak secondaries -> sequential fallback
        return "sequential"

    def _estimate_complexity(self, query: str, context: Optional[Dict]) -> str:
        """Estimate query complexity: simple, medium, hard."""
        complexity_indicators = {
            "simple": ["what is", "define", "list", "show", "get", "simple"],
            "medium": ["how to", "explain", "compare", "difference between", "why"],
            "hard": ["analyze", "optimize", "design", "prove", "complex", "architecture",
                    "distributed", "scale", "performance", "benchmark"]
        }

        query_lower = query.lower()

        # Check for hard indicators
        hard_matches = sum(
            1 for indicator in complexity_indicators["hard"]
            if indicator in query_lower
        )
        if hard_matches >= 2 or len(query) > 200:
            return "hard"

        # Check for simple indicators
        simple_matches = sum(
            1 for indicator in complexity_indicators["simple"]
            if indicator in query_lower
        )
        if simple_matches >= 1 and len(query) < 50:
            return "simple"

        # Default to medium
        return "medium"

    def get_routing_statistics(self) -> Dict[str, int]:
        """Get routing statistics by domain."""
        return {domain.value: count for domain, count in self.routing_stats.items()}

    def reset_statistics(self):
        """Reset routing statistics."""
        self.routing_stats = {domain: 0 for domain in ExpertDomain}

    def explain_routing(self, decision: RoutingDecision) -> str:
        """Generate a human-readable explanation of the routing decision."""
        explanation = []

        explanation.append(f"Primary Expert: {decision.primary_expert.domain.value}")
        explanation.append(f"Confidence: {decision.primary_expert.confidence:.2f}")
        explanation.append(f"Reasoning: {decision.primary_expert.reasoning}")

        if decision.secondary_experts:
            explanation.append("\nSecondary Experts:")
            for expert in decision.secondary_experts:
                explanation.append(f"  - {expert.domain.value} (confidence: {expert.confidence:.2f})")

        explanation.append(f"\nRouting Strategy: {decision.routing_strategy}")
        explanation.append(f"Estimated Complexity: {decision.estimated_complexity}")

        if decision.should_ensemble:
            explanation.append("\nWill ensemble outputs from multiple experts")

        return "\n".join(explanation)


if __name__ == "__main__":
    # Test the router
    router = MoERouter(enable_multi_expert=True)

    test_queries = [
        "Write a function to optimize this SQL query",
        "Explain how to deploy this application to Kubernetes",
        "Find security vulnerabilities in this code",
        "Design a scalable microservices architecture",
        "Generate unit tests for this function",
        "Monitor and troubleshoot production issues",
        "Analyze the performance bottlenecks in this system",
    ]

    print("=" * 80)
    print("MoE Router Test Results")
    print("=" * 80)

    for query in test_queries:
        decision = router.route(query)
        print(f"\nQuery: {query}")
        print("-" * 80)
        print(router.explain_routing(decision))
        print()

    print("=" * 80)
    print("Routing Statistics:")
    print(json.dumps(router.get_routing_statistics(), indent=2))

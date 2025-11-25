#!/usr/bin/env python3
"""
RAG Planner - Query Decomposition and Strategy Selection

First phase of Deep-Thinking RAG pipeline:
- Decomposes complex queries into sub-queries
- Decides retrieval strategy (internal docs vs web search)
- Creates execution plan

Author: LAT5150DRVMIL AI Framework
Version: 1.0.0
"""

from typing import List, Dict, Tuple
import re


class RAGPlanner:
    """
    Plan RAG retrieval strategy and decompose queries

    Usage:
        planner = RAGPlanner()
        plan = planner.plan("How do I optimize PostgreSQL for read-heavy workloads?")
        # Returns: {
        #   "sub_queries": ["indexing strategies", "query optimization", "caching"],
        #   "retrieval_strategy": "hybrid",
        #   "needs_web_search": False
        # }
    """

    def __init__(self):
        """Initialize RAG planner"""

        # Multi-part query indicators
        self.multi_part_indicators = [
            r'\band\b',
            r'\bor\b',
            r'\bvs\.?\b',
            r'\bversus\b',
            r'\bcompare\b',
            r'\bdifference between\b',
            r'\bfirst.*then\b',
            r'\bstep[- ]by[- ]step\b'
        ]

        # Web search indicators
        self.web_search_indicators = [
            'latest', 'recent', 'current', 'today', 'this week',
            'news', 'what happened', 'update'
        ]

        # Complex query indicators
        self.complex_indicators = [
            'comprehensive', 'detailed', 'in-depth',
            'analyze', 'investigate', 'explore'
        ]

    def plan(self, query: str) -> Dict:
        """
        Create execution plan for query

        Args:
            query: User query

        Returns:
            Plan dict with sub_queries, strategy, needs_web_search
        """
        query_lower = query.lower()

        # Detect if multi-part query
        is_multi_part = self._detect_multi_part(query_lower)

        # Decompose query
        if is_multi_part:
            sub_queries = self._decompose_query(query)
        else:
            sub_queries = [query]  # Single query

        # Decide retrieval strategy
        retrieval_strategy = self._select_strategy(query_lower)

        # Check if web search needed
        needs_web_search = self._needs_web_search(query_lower)

        return {
            "sub_queries": sub_queries,
            "retrieval_strategy": retrieval_strategy,
            "needs_web_search": needs_web_search,
            "is_complex": is_multi_part or any(ind in query_lower for ind in self.complex_indicators)
        }

    def _detect_multi_part(self, query: str) -> bool:
        """Detect if query has multiple parts"""
        for pattern in self.multi_part_indicators:
            if re.search(pattern, query):
                return True
        return False

    def _decompose_query(self, query: str) -> List[str]:
        """
        Decompose complex query into sub-queries

        Strategy:
        - Split on conjunctions (and, or)
        - Split on comparison terms (vs, versus, compare)
        - Split on sequential terms (first...then)
        """
        query_lower = query.lower()
        sub_queries = []

        # Strategy 1: Split on "and"
        if ' and ' in query_lower:
            parts = re.split(r'\s+and\s+', query, flags=re.IGNORECASE)
            sub_queries.extend([p.strip() for p in parts if p.strip()])

        # Strategy 2: Split on "vs" or "versus"
        elif re.search(r'\bvs\.?\b|\bversus\b', query_lower):
            parts = re.split(r'\s+vs\.?\s+|\s+versus\s+', query, flags=re.IGNORECASE)
            sub_queries.extend([p.strip() for p in parts if p.strip()])

        # Strategy 3: Comparison queries
        elif 'difference between' in query_lower or 'compare' in query_lower:
            # Extract compared items
            match = re.search(r'(?:difference between|compare)\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+)', query, re.IGNORECASE)
            if match:
                sub_queries.append(match.group(1).strip())
                sub_queries.append(match.group(2).strip())
            else:
                sub_queries = [query]

        # Strategy 4: Sequential queries (first...then)
        elif 'first' in query_lower and 'then' in query_lower:
            parts = re.split(r'\s+then\s+', query, flags=re.IGNORECASE)
            sub_queries.extend([p.strip().replace('first ', '') for p in parts if p.strip()])

        else:
            # No clear decomposition strategy, use whole query
            sub_queries = [query]

        # Deduplicate and clean
        sub_queries = list(dict.fromkeys(sub_queries))  # Preserve order
        sub_queries = [q for q in sub_queries if len(q) > 3]  # Filter too short

        return sub_queries if sub_queries else [query]

    def _select_strategy(self, query: str) -> str:
        """
        Select retrieval strategy

        Returns:
            "vector", "keyword", or "hybrid"
        """
        # Use vector for semantic queries
        semantic_indicators = ['how', 'why', 'explain', 'what is', 'describe']
        has_semantic = any(ind in query for ind in semantic_indicators)

        # Use keyword for specific terms
        keyword_indicators = ['exact', 'specific', 'named', 'called']
        has_keyword = any(ind in query for ind in keyword_indicators)

        # Default to hybrid
        if has_semantic and not has_keyword:
            return "vector"
        elif has_keyword and not has_semantic:
            return "keyword"
        else:
            return "hybrid"

    def _needs_web_search(self, query: str) -> bool:
        """Check if query needs web search"""
        return any(ind in query for ind in self.web_search_indicators)

    def estimate_complexity(self, query: str) -> Tuple[str, float]:
        """
        Estimate query complexity

        Returns:
            Tuple of (complexity_level, confidence)
            complexity_level: "simple", "medium", "complex"
        """
        query_lower = query.lower()

        complexity_score = 0

        # Multi-part adds complexity
        if self._detect_multi_part(query_lower):
            complexity_score += 2

        # Complex indicators
        for ind in self.complex_indicators:
            if ind in query_lower:
                complexity_score += 1

        # Length
        word_count = len(query.split())
        if word_count > 20:
            complexity_score += 1
        elif word_count > 10:
            complexity_score += 0.5

        # Technical terms
        technical_terms = ['algorithm', 'optimize', 'architecture', 'distributed', 'performance']
        for term in technical_terms:
            if term in query_lower:
                complexity_score += 0.5

        # Classify
        if complexity_score < 1:
            return "simple", 0.8
        elif complexity_score < 3:
            return "medium", 0.7
        else:
            return "complex", 0.9


if __name__ == "__main__":
    # Demo usage
    print("="*70)
    print("RAG Planner Demo")
    print("="*70)

    planner = RAGPlanner()

    test_queries = [
        "What is PostgreSQL?",
        "How do I optimize PostgreSQL and configure connection pooling?",
        "Compare PostgreSQL vs MySQL for production workloads",
        "Explain the difference between B-tree and GIN indexes",
        "First create indexes, then run EXPLAIN ANALYZE",
        "Analyze comprehensive PostgreSQL performance tuning strategies"
    ]

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print("-"*70)

        plan = planner.plan(query)
        complexity, confidence = planner.estimate_complexity(query)

        print(f"\nPlan:")
        print(f"  Sub-queries ({len(plan['sub_queries'])}):")
        for i, sq in enumerate(plan['sub_queries'], 1):
            print(f"    {i}. {sq}")
        print(f"  Retrieval strategy: {plan['retrieval_strategy']}")
        print(f"  Needs web search: {plan['needs_web_search']}")
        print(f"  Is complex: {plan['is_complex']}")
        print(f"\nComplexity: {complexity} (confidence: {confidence:.2f})")

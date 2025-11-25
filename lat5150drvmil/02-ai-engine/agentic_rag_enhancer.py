#!/usr/bin/env python3
"""
Agentic RAG Enhancer
Based on ai-that-works Episode #28: "Agentic RAG"

Key Capabilities:
- Agent-driven query reformulation for better retrieval
- Multi-hop retrieval (iterative query expansion)
- Source credibility scoring
- Adaptive retrieval strategies based on query type
- Query intent analysis

Benefits:
- Higher quality retrieval results
- Better handling of complex queries
- Source reliability assessment
- Intelligent query decomposition
"""

import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class QueryIntent(Enum):
    """Query intent categories"""
    FACTUAL = "factual"  # Seeking specific facts
    ANALYTICAL = "analytical"  # Requires analysis
    COMPARISON = "comparison"  # Comparing multiple things
    PROCEDURAL = "procedural"  # How-to or step-by-step
    EXPLORATORY = "exploratory"  # Broad topic exploration
    TEMPORAL = "temporal"  # Time-sensitive queries
    UNKNOWN = "unknown"


class RetrievalStrategy(Enum):
    """Retrieval strategies for different query types"""
    SINGLE_PASS = "single_pass"  # One retrieval, good enough
    MULTI_HOP = "multi_hop"  # Iterative retrieval
    HYBRID = "hybrid"  # Combine multiple approaches
    DECOMPOSED = "decomposed"  # Break into sub-queries


@dataclass
class ReformulatedQuery:
    """
    Reformulated query with metadata

    Attributes:
        original: Original user query
        reformulated: Improved query for RAG
        intent: Detected intent
        strategy: Recommended retrieval strategy
        sub_queries: Optional sub-queries for decomposition
        reasoning: Why this reformulation
    """
    original: str
    reformulated: str
    intent: QueryIntent
    strategy: RetrievalStrategy
    sub_queries: List[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class CredibilityScore:
    """
    Source credibility assessment

    Attributes:
        score: Overall credibility (0.0-1.0)
        recency: How recent is the information
        authority: Is source authoritative
        consistency: Consistent with other sources
        reasoning: Explanation of score
    """
    score: float
    recency: float = 0.5
    authority: float = 0.5
    consistency: float = 0.5
    reasoning: str = ""


@dataclass
class AgenticRetrievalResult:
    """
    Enhanced retrieval result with agentic metadata

    Attributes:
        chunks: Retrieved text chunks
        credibility: Source credibility scores
        query_reformulation: How query was reformulated
        hops: Number of retrieval hops performed
        strategy: Strategy used
        reasoning: Explanation of retrieval process
    """
    chunks: List[Dict[str, Any]]
    credibility: List[CredibilityScore]
    query_reformulation: ReformulatedQuery
    hops: int
    strategy: RetrievalStrategy
    reasoning: str


class AgenticRAGEnhancer:
    """
    Enhance RAG system with agentic capabilities

    Use Cases:
    - Complex queries requiring query decomposition
    - Multi-step reasoning over documents
    - Source reliability assessment
    - Adaptive retrieval based on query type
    """

    def __init__(self, rag_system, llm_engine=None):
        """
        Initialize agentic RAG enhancer

        Args:
            rag_system: Underlying RAG system (EnhancedRAGSystem)
            llm_engine: Optional LLM for query reformulation
        """
        self.rag_system = rag_system
        self.llm_engine = llm_engine
        self.query_history: List[ReformulatedQuery] = []
        self.credibility_cache: Dict[str, CredibilityScore] = {}

    def query(
        self,
        user_query: str,
        max_hops: int = 3,
        top_k: int = 5,
        enable_reformulation: bool = True,
        enable_credibility: bool = True
    ) -> AgenticRetrievalResult:
        """
        Agentic RAG query with intelligent processing

        Args:
            user_query: User's query
            max_hops: Maximum retrieval hops for multi-hop
            top_k: Number of results per hop
            enable_reformulation: Enable query reformulation
            enable_credibility: Enable credibility scoring

        Returns:
            AgenticRetrievalResult with enhanced metadata
        """
        # Step 1: Analyze query intent and reformulate
        if enable_reformulation:
            reformulated = self.reformulate_query(user_query)
        else:
            reformulated = ReformulatedQuery(
                original=user_query,
                reformulated=user_query,
                intent=QueryIntent.UNKNOWN,
                strategy=RetrievalStrategy.SINGLE_PASS
            )

        # Step 2: Choose retrieval strategy based on intent
        chunks, hops, reasoning = self._execute_retrieval_strategy(
            reformulated,
            max_hops=max_hops,
            top_k=top_k
        )

        # Step 3: Score credibility of sources
        credibility_scores = []
        if enable_credibility:
            for chunk in chunks:
                score = self.score_credibility(chunk)
                credibility_scores.append(score)

        # Store in history
        self.query_history.append(reformulated)

        return AgenticRetrievalResult(
            chunks=chunks,
            credibility=credibility_scores,
            query_reformulation=reformulated,
            hops=hops,
            strategy=reformulated.strategy,
            reasoning=reasoning
        )

    def reformulate_query(self, user_query: str) -> ReformulatedQuery:
        """
        Reformulate query for better retrieval

        Args:
            user_query: Original user query

        Returns:
            ReformulatedQuery with improved query and metadata
        """
        # Step 1: Detect intent
        intent = self._detect_intent(user_query)

        # Step 2: Choose strategy
        strategy = self._choose_strategy(intent, user_query)

        # Step 3: Reformulate based on intent
        if self.llm_engine:
            # Use LLM for sophisticated reformulation
            reformulated, sub_queries, reasoning = self._llm_reformulate(
                user_query,
                intent,
                strategy
            )
        else:
            # Use heuristic reformulation
            reformulated, sub_queries, reasoning = self._heuristic_reformulate(
                user_query,
                intent,
                strategy
            )

        return ReformulatedQuery(
            original=user_query,
            reformulated=reformulated,
            intent=intent,
            strategy=strategy,
            sub_queries=sub_queries,
            reasoning=reasoning
        )

    def _detect_intent(self, query: str) -> QueryIntent:
        """
        Detect query intent using pattern matching

        Args:
            query: User query

        Returns:
            QueryIntent classification
        """
        query_lower = query.lower()

        # Patterns for different intents
        patterns = {
            QueryIntent.COMPARISON: [
                r"compar(e|ison)",
                r"(difference|differ) between",
                r"vs\.?|versus",
                r"better than",
                r"(advantage|disadvantage)s? of"
            ],
            QueryIntent.PROCEDURAL: [
                r"how (do|to|can)",
                r"step[s]? (to|for)",
                r"guide (for|to)",
                r"tutorial",
                r"instructions?"
            ],
            QueryIntent.TEMPORAL: [
                r"latest|recent|newest",
                r"current|now",
                r"(update|change)s?",
                r"as of|since",
                r"(\d{4}|today|yesterday)"
            ],
            QueryIntent.ANALYTICAL: [
                r"why|explain|analyze",
                r"reason(s|ing)?",
                r"cause(s)?",
                r"impact|effect",
                r"implication"
            ],
            QueryIntent.FACTUAL: [
                r"^what is",
                r"^who is",
                r"^when (did|was)",
                r"^where (is|was)",
                r"definition"
            ]
        }

        # Check patterns
        for intent, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, query_lower):
                    return intent

        # Default to exploratory if query is long/complex
        if len(query.split()) > 10:
            return QueryIntent.EXPLORATORY

        return QueryIntent.FACTUAL

    def _choose_strategy(self, intent: QueryIntent, query: str) -> RetrievalStrategy:
        """
        Choose retrieval strategy based on intent

        Args:
            intent: Detected query intent
            query: Original query

        Returns:
            Recommended retrieval strategy
        """
        # Strategy mapping
        strategy_map = {
            QueryIntent.FACTUAL: RetrievalStrategy.SINGLE_PASS,
            QueryIntent.COMPARISON: RetrievalStrategy.DECOMPOSED,
            QueryIntent.PROCEDURAL: RetrievalStrategy.MULTI_HOP,
            QueryIntent.ANALYTICAL: RetrievalStrategy.MULTI_HOP,
            QueryIntent.EXPLORATORY: RetrievalStrategy.HYBRID,
            QueryIntent.TEMPORAL: RetrievalStrategy.SINGLE_PASS
        }

        # Complex queries benefit from multi-hop
        if len(query.split()) > 15:
            return RetrievalStrategy.MULTI_HOP

        return strategy_map.get(intent, RetrievalStrategy.SINGLE_PASS)

    def _heuristic_reformulate(
        self,
        query: str,
        intent: QueryIntent,
        strategy: RetrievalStrategy
    ) -> Tuple[str, List[str], str]:
        """
        Heuristic query reformulation without LLM

        Args:
            query: Original query
            intent: Detected intent
            strategy: Retrieval strategy

        Returns:
            Tuple of (reformulated_query, sub_queries, reasoning)
        """
        sub_queries = []
        reasoning = f"Intent: {intent.value}, Strategy: {strategy.value}"

        # For comparisons, decompose into sub-queries
        if intent == QueryIntent.COMPARISON:
            # Extract entities being compared
            parts = re.split(r" vs\.? | versus | compared to | difference between ", query, flags=re.IGNORECASE)
            if len(parts) >= 2:
                sub_queries = [f"What is {part.strip()}?" for part in parts[:2]]
                reasoning += " | Decomposed comparison into entity queries"

        # For procedural queries, make them more specific
        elif intent == QueryIntent.PROCEDURAL:
            # Add context words
            if "how to" in query.lower():
                reformulated = query + " step by step guide tutorial"
                return reformulated, sub_queries, reasoning + " | Added procedural keywords"

        # For temporal queries, emphasize recency
        elif intent == QueryIntent.TEMPORAL:
            reformulated = "latest current " + query
            return reformulated, sub_queries, reasoning + " | Emphasized recency"

        # Default: return original
        return query, sub_queries, reasoning

    def _llm_reformulate(
        self,
        query: str,
        intent: QueryIntent,
        strategy: RetrievalStrategy
    ) -> Tuple[str, List[str], str]:
        """
        LLM-based query reformulation

        Args:
            query: Original query
            intent: Detected intent
            strategy: Retrieval strategy

        Returns:
            Tuple of (reformulated_query, sub_queries, reasoning)
        """
        prompt = f"""Reformulate this query for better document retrieval.

Original query: "{query}"
Intent: {intent.value}
Strategy: {strategy.value}

Tasks:
1. Improve the query for better keyword matching
2. If strategy is "decomposed", break into 2-3 sub-queries
3. Explain your reasoning

Format:
REFORMULATED: <improved query>
SUB_QUERIES: <query 1> | <query 2> | ...
REASONING: <explanation>"""

        try:
            # Query LLM
            response = self._query_llm(prompt)

            # Parse response
            reformulated = query  # Default fallback
            sub_queries = []
            reasoning = "LLM reformulation"

            # Extract reformulated query
            ref_match = re.search(r"REFORMULATED:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
            if ref_match:
                reformulated = ref_match.group(1).strip()

            # Extract sub-queries
            sub_match = re.search(r"SUB_QUERIES:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
            if sub_match:
                sub_queries = [q.strip() for q in sub_match.group(1).split("|")]

            # Extract reasoning
            reas_match = re.search(r"REASONING:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
            if reas_match:
                reasoning = reas_match.group(1).strip()

            return reformulated, sub_queries, reasoning

        except Exception as e:
            # Fallback to heuristic
            return self._heuristic_reformulate(query, intent, strategy)

    def _execute_retrieval_strategy(
        self,
        reformulated: ReformulatedQuery,
        max_hops: int,
        top_k: int
    ) -> Tuple[List[Dict], int, str]:
        """
        Execute retrieval based on chosen strategy

        Args:
            reformulated: Reformulated query
            max_hops: Maximum hops for multi-hop
            top_k: Results per hop

        Returns:
            Tuple of (chunks, hops_used, reasoning)
        """
        strategy = reformulated.strategy

        if strategy == RetrievalStrategy.SINGLE_PASS:
            # Simple single retrieval
            results = self.rag_system.query(reformulated.reformulated, top_k=top_k)
            return results, 1, "Single-pass retrieval"

        elif strategy == RetrievalStrategy.DECOMPOSED and reformulated.sub_queries:
            # Query each sub-query separately
            all_results = []
            for sub_query in reformulated.sub_queries:
                sub_results = self.rag_system.query(sub_query, top_k=top_k // len(reformulated.sub_queries))
                all_results.extend(sub_results)

            # Deduplicate by chunk ID
            seen = set()
            unique_results = []
            for result in all_results:
                chunk_id = result.get('chunk_id', result.get('text', '')[:50])
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    unique_results.append(result)

            return unique_results[:top_k], len(reformulated.sub_queries), "Decomposed into sub-queries"

        elif strategy == RetrievalStrategy.MULTI_HOP:
            # Iterative retrieval with query expansion
            return self._multi_hop_retrieval(reformulated.reformulated, max_hops, top_k)

        else:  # HYBRID
            # Combine single-pass with query expansion
            initial_results = self.rag_system.query(reformulated.reformulated, top_k=top_k)

            if max_hops > 1:
                # Extract key terms for expansion
                expanded_results, hops_used = self._expand_from_results(
                    initial_results,
                    reformulated.reformulated,
                    max_hops - 1,
                    top_k
                )
                all_results = initial_results + expanded_results

                # Deduplicate
                seen = set()
                unique = []
                for r in all_results:
                    chunk_id = r.get('chunk_id', r.get('text', '')[:50])
                    if chunk_id not in seen:
                        seen.add(chunk_id)
                        unique.append(r)

                return unique[:top_k], hops_used + 1, "Hybrid: initial + expansion"
            else:
                return initial_results, 1, "Hybrid (single hop only)"

    def _multi_hop_retrieval(
        self,
        query: str,
        max_hops: int,
        top_k: int
    ) -> Tuple[List[Dict], int, str]:
        """
        Multi-hop retrieval with iterative query expansion

        Args:
            query: Initial query
            max_hops: Maximum hops
            top_k: Results per hop

        Returns:
            Tuple of (all_results, hops_used, reasoning)
        """
        all_results = []
        seen_chunks = set()
        current_query = query

        for hop in range(max_hops):
            # Retrieve with current query
            hop_results = self.rag_system.query(current_query, top_k=top_k)

            # Add new results
            new_results = []
            for result in hop_results:
                chunk_id = result.get('chunk_id', result.get('text', '')[:50])
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    new_results.append(result)
                    all_results.append(result)

            # Stop if no new results
            if not new_results:
                return all_results, hop + 1, f"Multi-hop stopped at hop {hop + 1} (no new results)"

            # Generate next query from results
            if hop < max_hops - 1:
                current_query = self._generate_expansion_query(new_results, query)

        return all_results, max_hops, f"Multi-hop completed {max_hops} hops"

    def _expand_from_results(
        self,
        results: List[Dict],
        original_query: str,
        hops: int,
        top_k: int
    ) -> Tuple[List[Dict], int]:
        """Expand retrieval from initial results"""
        expansion_query = self._generate_expansion_query(results, original_query)
        expanded, hops_used, _ = self._multi_hop_retrieval(expansion_query, hops, top_k)
        return expanded, hops_used

    def _generate_expansion_query(self, results: List[Dict], original_query: str) -> str:
        """
        Generate expansion query from results

        Args:
            results: Current retrieval results
            original_query: Original user query

        Returns:
            Expanded query string
        """
        # Extract key terms from top result
        if not results:
            return original_query

        top_text = results[0].get('chunk_text', results[0].get('text', ''))

        # Extract capitalized terms (likely entities/concepts)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', top_text)

        if entities:
            # Combine original query with top entities
            return f"{original_query} {' '.join(entities[:3])}"

        return original_query

    def score_credibility(self, chunk: Dict[str, Any]) -> CredibilityScore:
        """
        Score source credibility

        Args:
            chunk: Retrieved chunk with metadata

        Returns:
            CredibilityScore assessment
        """
        # Check cache
        chunk_id = chunk.get('chunk_id', '')
        if chunk_id in self.credibility_cache:
            return self.credibility_cache[chunk_id]

        # Default scores
        recency = 0.5
        authority = 0.5
        consistency = 0.5
        reasoning = []

        # Check metadata for signals
        metadata = chunk.get('metadata', {})

        # Recency scoring
        if 'timestamp' in metadata:
            # Recent content scores higher
            age_days = (datetime.now() - metadata['timestamp']).days
            recency = max(0.1, 1.0 - (age_days / 365))  # Decay over year
            reasoning.append(f"Age: {age_days} days")

        # Authority scoring
        source = metadata.get('source', '')
        if any(term in source.lower() for term in ['official', 'documentation', 'paper', 'publication']):
            authority = 0.9
            reasoning.append("Authoritative source")
        elif any(term in source.lower() for term in ['blog', 'forum', 'comment']):
            authority = 0.3
            reasoning.append("Informal source")

        # Overall score (weighted average)
        score = (recency * 0.3) + (authority * 0.5) + (consistency * 0.2)

        credibility = CredibilityScore(
            score=score,
            recency=recency,
            authority=authority,
            consistency=consistency,
            reasoning="; ".join(reasoning) if reasoning else "Default scoring"
        )

        # Cache result
        if chunk_id:
            self.credibility_cache[chunk_id] = credibility

        return credibility

    def _query_llm(self, prompt: str) -> str:
        """Query LLM engine"""
        if hasattr(self.llm_engine, 'query'):
            result = self.llm_engine.query(prompt, use_rag=False, use_cache=False)
            return result.content if hasattr(result, 'content') else str(result)
        elif hasattr(self.llm_engine, 'generate'):
            result = self.llm_engine.generate(prompt)
            return result.get('response', '')
        else:
            raise ValueError("LLM engine does not have query() or generate() method")

    def get_statistics(self) -> Dict[str, Any]:
        """Get agentic RAG statistics"""
        intent_counts = {}
        for query in self.query_history:
            intent = query.intent.value
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

        strategy_counts = {}
        for query in self.query_history:
            strategy = query.strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        return {
            "total_queries": len(self.query_history),
            "intent_distribution": intent_counts,
            "strategy_distribution": strategy_counts,
            "credibility_cache_size": len(self.credibility_cache),
            "avg_sub_queries": sum(len(q.sub_queries) for q in self.query_history) / max(len(self.query_history), 1)
        }


def main():
    """Demo usage"""
    print("=== Agentic RAG Enhancer Demo ===\n")

    # Mock RAG system
    class MockRAG:
        def query(self, query: str, top_k: int = 5):
            return [
                {
                    'chunk_id': f'chunk_{i}',
                    'text': f'Mock result {i} for query: {query}',
                    'chunk_text': f'This is mock content about {query}. Entity Name and concept.',
                    'metadata': {'source': 'documentation'}
                }
                for i in range(top_k)
            ]

    rag = MockRAG()
    enhancer = AgenticRAGEnhancer(rag)

    # Test queries
    test_queries = [
        "What is Python?",
        "Compare Python vs JavaScript",
        "How to set up a web server",
        "Latest developments in AI"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        result = enhancer.query(query, max_hops=2, top_k=3)

        print(f"  Intent: {result.query_reformulation.intent.value}")
        print(f"  Strategy: {result.strategy.value}")
        print(f"  Reformulated: {result.query_reformulation.reformulated}")
        print(f"  Hops: {result.hops}")
        print(f"  Results: {len(result.chunks)}")

        if result.credibility:
            avg_cred = sum(c.score for c in result.credibility) / len(result.credibility)
            print(f"  Avg Credibility: {avg_cred:.2f}")

    # Statistics
    print("\n\nStatistics:")
    stats = enhancer.get_statistics()
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Intent distribution: {stats['intent_distribution']}")
    print(f"  Strategy distribution: {stats['strategy_distribution']}")


if __name__ == "__main__":
    main()

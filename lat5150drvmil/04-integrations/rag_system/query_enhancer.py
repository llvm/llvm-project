#!/usr/bin/env python3
"""
Query Enhancement Module for Screenshot Intelligence RAG System

Implements:
- Synonym expansion for better recall (+3-5% expected gain)
- LLM-based query rewriting for complex queries (+5-7% expected gain)
- Query context enrichment
- Multi-query generation for comprehensive retrieval

Based on research:
- Lai et al. (2023): Query expansion improves recall 3-7%
- Databricks (2024): Rewriting improves precision on complex queries
"""

import logging
import re
from typing import List, Dict, Optional, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Common synonym mappings for technical/error domains
SYNONYM_DICT = {
    # Error-related terms
    'error': ['fail', 'failure', 'exception', 'crash', 'problem', 'issue'],
    'fail': ['error', 'failure', 'crash', 'problem'],
    'crash': ['error', 'fail', 'hang', 'freeze', 'terminated'],
    'bug': ['error', 'issue', 'problem', 'defect', 'glitch'],

    # Network terms
    'network': ['connection', 'internet', 'wifi', 'connectivity', 'lan'],
    'connection': ['network', 'connect', 'link', 'session'],
    'timeout': ['delay', 'slow', 'hang', 'unresponsive'],
    'vpn': ['virtual private network', 'tunnel', 'proxy'],

    # System terms
    'memory': ['ram', 'heap', 'allocation', 'oom'],
    'disk': ['storage', 'drive', 'filesystem', 'volume'],
    'cpu': ['processor', 'core', 'thread', 'process'],
    'load': ['usage', 'utilization', 'consumption', 'performance'],

    # Application terms
    'app': ['application', 'program', 'software', 'service'],
    'service': ['daemon', 'process', 'server', 'app'],
    'restart': ['reboot', 'reload', 'refresh', 'reset'],
    'install': ['setup', 'deploy', 'configuration', 'installation'],

    # Authentication terms
    'login': ['signin', 'authentication', 'auth', 'access'],
    'password': ['credential', 'passphrase', 'auth', 'authentication'],
    'permission': ['access', 'authorization', 'privilege', 'rights'],

    # Status terms
    'slow': ['lag', 'latency', 'delay', 'performance', 'timeout'],
    'fast': ['quick', 'rapid', 'speed', 'performance'],
    'unavailable': ['down', 'offline', 'unreachable', 'inaccessible'],
    'working': ['functional', 'operational', 'running', 'active'],
}


@dataclass
class EnhancedQuery:
    """Enhanced query with expansions"""
    original: str
    expanded: str
    synonyms_added: List[str]
    method: str  # 'synonym', 'llm', 'hybrid'


class QueryEnhancer:
    """
    Query enhancement for improved retrieval accuracy

    Features:
    - Synonym expansion (domain-specific)
    - Stop word preservation for technical queries
    - Multi-query generation
    - LLM-based rewriting (optional)
    """

    def __init__(self, use_llm: bool = False, llm_endpoint: Optional[str] = None):
        """
        Initialize query enhancer

        Args:
            use_llm: Enable LLM-based query rewriting
            llm_endpoint: Ollama endpoint (e.g., 'http://localhost:11434')
        """
        self.use_llm = use_llm
        self.llm_endpoint = llm_endpoint or "http://localhost:11434"
        self.synonym_dict = SYNONYM_DICT

        if use_llm:
            logger.info("Query enhancer initialized with LLM rewriting")
        else:
            logger.info("Query enhancer initialized with synonym expansion only")

    def enhance_query(self, query: str, max_synonyms: int = 3) -> EnhancedQuery:
        """
        Enhance query with synonym expansion

        Args:
            query: Original search query
            max_synonyms: Maximum synonyms to add per term

        Returns:
            EnhancedQuery with expanded terms
        """
        # Tokenize query (preserve technical terms)
        tokens = self._tokenize(query)

        # Find synonyms
        expanded_tokens = []
        synonyms_added = []

        for token in tokens:
            expanded_tokens.append(token)

            # Look up synonyms
            token_lower = token.lower()
            if token_lower in self.synonym_dict:
                syns = self.synonym_dict[token_lower][:max_synonyms]
                expanded_tokens.extend(syns)
                synonyms_added.extend(syns)

        # Build expanded query
        expanded = ' '.join(expanded_tokens)

        return EnhancedQuery(
            original=query,
            expanded=expanded,
            synonyms_added=synonyms_added,
            method='synonym'
        )

    def enhance_with_llm(self, query: str, context: Optional[str] = None) -> EnhancedQuery:
        """
        Enhance query using LLM rewriting

        Args:
            query: Original search query
            context: Optional context to help rewriting

        Returns:
            EnhancedQuery with LLM-rewritten query
        """
        if not self.use_llm:
            logger.warning("LLM not enabled, falling back to synonym expansion")
            return self.enhance_query(query)

        # Build LLM prompt
        prompt = self._build_rewrite_prompt(query, context)

        # Call LLM
        try:
            rewritten = self._call_ollama(prompt)

            return EnhancedQuery(
                original=query,
                expanded=rewritten,
                synonyms_added=[],
                method='llm'
            )
        except Exception as e:
            logger.error(f"LLM rewriting failed: {e}, using synonym expansion")
            return self.enhance_query(query)

    def generate_multi_queries(self, query: str, num_queries: int = 3) -> List[str]:
        """
        Generate multiple query variations for comprehensive retrieval

        Args:
            query: Original query
            num_queries: Number of variations to generate

        Returns:
            List of query variations
        """
        queries = [query]  # Original

        # Add synonym-expanded version
        enhanced = self.enhance_query(query, max_synonyms=2)
        queries.append(enhanced.expanded)

        # Add keyword-focused version (remove filler words)
        keywords = self._extract_keywords(query)
        queries.append(' '.join(keywords))

        return queries[:num_queries]

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text preserving technical terms"""
        # Split on whitespace but preserve alphanumeric+hyphen+underscore
        tokens = re.findall(r'\w[\w\-]*', text)
        return tokens

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords by removing common filler words"""
        # Common filler words (but preserve technical terms)
        fillers = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'can', 'may', 'might', 'must',
            'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'this', 'that', 'these', 'those',
            'what', 'which', 'who', 'when', 'where', 'why', 'how',
            'in', 'on', 'at', 'to', 'for', 'of', 'from', 'by', 'with',
            'about', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'over'
        }

        tokens = self._tokenize(query)
        keywords = [t for t in tokens if t.lower() not in fillers]

        return keywords

    def _build_rewrite_prompt(self, query: str, context: Optional[str] = None) -> str:
        """Build prompt for LLM query rewriting"""
        base_prompt = f"""You are helping improve search queries for a technical screenshot analysis system.

Original query: "{query}"
"""

        if context:
            base_prompt += f"\nContext: {context}\n"

        base_prompt += """
Task: Rewrite this query to be more effective for semantic search. Add relevant technical terms, synonyms, and context that would help find related screenshots, error messages, or system events.

Guidelines:
1. Expand abbreviations (e.g., VPN -> Virtual Private Network)
2. Add common synonyms for key terms
3. Include related technical concepts
4. Keep it concise (1-2 sentences max)
5. Focus on terms that would appear in screenshots or logs

Rewritten query:"""

        return base_prompt

    def _call_ollama(self, prompt: str, model: str = "llama3.2:3b") -> str:
        """Call Ollama API for query rewriting"""
        import requests

        try:
            response = requests.post(
                f"{self.llm_endpoint}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Low temperature for consistency
                        "top_p": 0.9,
                        "max_tokens": 100
                    }
                },
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                rewritten = result.get('response', '').strip()

                # Extract just the rewritten query (remove any explanations)
                lines = rewritten.split('\n')
                return lines[0] if lines else rewritten
            else:
                raise Exception(f"Ollama API error: {response.status_code}")

        except Exception as e:
            raise Exception(f"Failed to call Ollama: {e}")


class HybridQueryProcessor:
    """
    Hybrid query processing combining multiple enhancement techniques

    Strategy:
    1. For simple queries (1-2 words): Use synonym expansion only
    2. For complex queries (3+ words): Use LLM rewriting + synonyms
    3. For technical queries (contains code/paths): Preserve exact terms
    """

    def __init__(self, use_llm: bool = False, llm_endpoint: Optional[str] = None):
        """Initialize hybrid processor"""
        self.enhancer = QueryEnhancer(use_llm=use_llm, llm_endpoint=llm_endpoint)

    def process(self, query: str) -> EnhancedQuery:
        """
        Process query using optimal strategy

        Args:
            query: Search query

        Returns:
            EnhancedQuery with best enhancement method
        """
        # Detect query type
        query_type = self._classify_query(query)

        if query_type == 'simple':
            # Simple: synonym expansion only
            return self.enhancer.enhance_query(query, max_synonyms=3)

        elif query_type == 'technical':
            # Technical: minimal expansion (preserve exact terms)
            return self.enhancer.enhance_query(query, max_synonyms=1)

        else:  # 'complex'
            # Complex: use LLM if available
            if self.enhancer.use_llm:
                return self.enhancer.enhance_with_llm(query)
            else:
                return self.enhancer.enhance_query(query, max_synonyms=3)

    def _classify_query(self, query: str) -> str:
        """
        Classify query type

        Returns:
            'simple', 'technical', or 'complex'
        """
        tokens = query.split()

        # Check for technical indicators
        technical_patterns = [
            r'/[\w/]+',  # File paths
            r'\w+\.\w+\.\w+',  # Domains or versions
            r'[A-Z_]{3,}',  # Constants
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',  # IP addresses
            r'0x[0-9a-fA-F]+',  # Hex values
            r'[a-z]+_[a-z]+',  # Snake_case identifiers
        ]

        for pattern in technical_patterns:
            if re.search(pattern, query):
                return 'technical'

        # Check query complexity
        if len(tokens) <= 2:
            return 'simple'
        else:
            return 'complex'


# Example usage and testing
if __name__ == "__main__":
    print("=== Query Enhancer Test ===\n")

    # Initialize enhancer
    enhancer = QueryEnhancer(use_llm=False)

    # Test queries
    test_queries = [
        "VPN connection error",
        "slow network performance",
        "app crash memory leak",
        "login failed authentication",
        "disk full storage",
    ]

    print("Synonym Expansion Tests:\n")
    for query in test_queries:
        enhanced = enhancer.enhance_query(query, max_synonyms=3)
        print(f"Original:  {enhanced.original}")
        print(f"Expanded:  {enhanced.expanded}")
        print(f"Synonyms:  {', '.join(enhanced.synonyms_added)}")
        print()

    # Test multi-query generation
    print("\nMulti-Query Generation Test:\n")
    query = "VPN connection timeout error"
    queries = enhancer.generate_multi_queries(query, num_queries=3)
    for i, q in enumerate(queries, 1):
        print(f"{i}. {q}")

    # Test hybrid processor
    print("\n\nHybrid Processor Tests:\n")
    processor = HybridQueryProcessor(use_llm=False)

    test_cases = [
        ("error", "simple keyword"),
        ("VPN connection failed", "complex query"),
        ("/var/log/syslog error", "technical query with path"),
        ("192.168.1.1 unreachable", "technical query with IP"),
    ]

    for query, description in test_cases:
        enhanced = processor.process(query)
        query_type = processor._classify_query(query)
        print(f"{description}: {query}")
        print(f"  Type:     {query_type}")
        print(f"  Method:   {enhanced.method}")
        print(f"  Expanded: {enhanced.expanded}")
        print()

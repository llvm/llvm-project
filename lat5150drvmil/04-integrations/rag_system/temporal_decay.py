#!/usr/bin/env python3
"""
Temporal Awareness for RAG System
Fixes "temporal blindness" by applying decay to time-sensitive documents

Problem: LLMs treat 10-month-old predictions as current data
Solution: Exponential decay based on document age and type
"""

import re
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class TemporalScope(Enum):
    """Document temporal scope classification"""
    MARKET_DATA = "market_data"        # Betting odds, stock prices, rates
    PREDICTIONS = "predictions"        # Forecasts, predictions, estimates
    NEWS = "news"                      # News articles, current events
    TECHNICAL_DOCS = "technical_docs"  # API docs, specifications
    EVERGREEN = "evergreen"            # Timeless content (algorithms, theory)


@dataclass
class TemporalProfile:
    """Decay profile for different document types"""
    scope: TemporalScope
    half_life_days: float
    stale_threshold_days: int
    requires_freshness: bool


# Temporal decay profiles
DECAY_PROFILES = {
    TemporalScope.MARKET_DATA: TemporalProfile(
        scope=TemporalScope.MARKET_DATA,
        half_life_days=7,      # Betting odds/prices decay rapidly
        stale_threshold_days=14,
        requires_freshness=True
    ),
    TemporalScope.PREDICTIONS: TemporalProfile(
        scope=TemporalScope.PREDICTIONS,
        half_life_days=30,     # Predictions decay over weeks
        stale_threshold_days=90,
        requires_freshness=True
    ),
    TemporalScope.NEWS: TemporalProfile(
        scope=TemporalScope.NEWS,
        half_life_days=14,     # News becomes old quickly
        stale_threshold_days=30,
        requires_freshness=True
    ),
    TemporalScope.TECHNICAL_DOCS: TemporalProfile(
        scope=TemporalScope.TECHNICAL_DOCS,
        half_life_days=365,    # Technical docs stable for months/years
        stale_threshold_days=730,
        requires_freshness=False
    ),
    TemporalScope.EVERGREEN: TemporalProfile(
        scope=TemporalScope.EVERGREEN,
        half_life_days=float('inf'),  # No decay
        stale_threshold_days=999999,
        requires_freshness=False
    ),
}


class DateExtractor:
    """Extract publication dates from documents"""

    # Common date patterns
    DATE_PATTERNS = [
        # ISO format: 2025-11-08, 2025/11/08
        (r'(\d{4})[-/](\d{2})[-/](\d{2})', '%Y-%m-%d'),
        # US format: 11/08/2025, 11-08-2025
        (r'(\d{2})[-/](\d{2})[-/](\d{4})', '%m-%d-%Y'),
        # UK format: 08/11/2025
        (r'(\d{2})[./](\d{2})[./](\d{4})', '%d.%m.%Y'),
        # Month name: November 8, 2025 or Nov 8, 2025
        (r'(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})', '%B %d %Y'),
        # 8 Nov 2025
        (r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})', '%d %b %Y'),
    ]

    # Metadata field names that contain dates
    DATE_FIELDS = [
        'date', 'publication_date', 'published', 'created', 'modified',
        'last_updated', 'timestamp', 'pub_date', 'created_at', 'updated_at'
    ]

    @staticmethod
    def extract_date(chunk: Dict) -> Optional[datetime]:
        """
        Extract publication date from chunk metadata or content

        Priority:
        1. Metadata fields (most reliable)
        2. Content extraction (less reliable)
        3. File modification time (fallback)
        """

        # Try metadata first
        metadata = chunk.get('metadata', {})

        # Check common date fields
        for field in DateExtractor.DATE_FIELDS:
            if field in metadata:
                date_str = metadata[field]
                if isinstance(date_str, str):
                    parsed = DateExtractor._parse_date_string(date_str)
                    if parsed:
                        return parsed
                elif isinstance(date_str, (int, float)):
                    # Unix timestamp
                    try:
                        return datetime.fromtimestamp(date_str)
                    except (ValueError, OSError):
                        pass

        # Try extracting from content
        content = chunk.get('text', '')
        extracted = DateExtractor._extract_from_text(content)
        if extracted:
            return extracted

        # Fallback to file modification time
        filename = metadata.get('filename', '')
        if filename:
            filepath = Path(filename)
            if filepath.exists():
                mtime = filepath.stat().st_mtime
                return datetime.fromtimestamp(mtime)

        # No date found - return None (will be treated as current)
        return None

    @staticmethod
    def _parse_date_string(date_str: str) -> Optional[datetime]:
        """Parse date string with multiple format attempts"""
        for pattern, fmt in DateExtractor.DATE_PATTERNS:
            match = re.search(pattern, date_str)
            if match:
                try:
                    return datetime.strptime(match.group(0), fmt)
                except ValueError:
                    continue
        return None

    @staticmethod
    def _extract_from_text(text: str) -> Optional[datetime]:
        """Extract most recent date from text content"""
        dates = []

        for pattern, fmt in DateExtractor.DATE_PATTERNS:
            for match in re.finditer(pattern, text):
                try:
                    date = datetime.strptime(match.group(0), fmt)
                    # Sanity check: date should be reasonable
                    if datetime(2000, 1, 1) <= date <= datetime.now() + timedelta(days=365):
                        dates.append(date)
                except ValueError:
                    continue

        # Return most recent date (likely publication date)
        return max(dates) if dates else None


class TemporalScopeClassifier:
    """Classify documents by temporal scope"""

    # Keywords for classification
    SCOPE_KEYWORDS = {
        TemporalScope.MARKET_DATA: [
            'odds', 'betting', 'price', 'stock', 'market', 'trading',
            'rate', 'exchange', 'forex', 'commodity', 'futures'
        ],
        TemporalScope.PREDICTIONS: [
            'predict', 'forecast', 'estimate', 'projection', 'outlook',
            'will be', 'expected to', 'likely to', 'probability'
        ],
        TemporalScope.NEWS: [
            'breaking', 'today', 'yesterday', 'reported', 'announced',
            'latest', 'update', 'developing', 'just in'
        ],
        TemporalScope.TECHNICAL_DOCS: [
            'documentation', 'api', 'reference', 'specification', 'manual',
            'guide', 'tutorial', 'readme', 'install', 'configuration'
        ],
        TemporalScope.EVERGREEN: [
            'algorithm', 'theory', 'mathematics', 'physics', 'principle',
            'fundamental', 'concept', 'definition', 'proof'
        ],
    }

    @staticmethod
    def classify(chunk: Dict) -> TemporalScope:
        """
        Classify chunk temporal scope

        Uses keyword matching + metadata hints
        """
        text = chunk.get('text', '').lower()
        metadata = chunk.get('metadata', {})
        filename = metadata.get('filename', '').lower()

        # Score each scope
        scores = {scope: 0 for scope in TemporalScope}

        for scope, keywords in TemporalScopeClassifier.SCOPE_KEYWORDS.items():
            for keyword in keywords:
                # Count keyword occurrences
                scores[scope] += text.count(keyword)
                scores[scope] += filename.count(keyword)

        # Get highest scoring scope
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)

        # Default: technical docs (most common in code repositories)
        return TemporalScope.TECHNICAL_DOCS


class TemporalAwareRetriever:
    """
    RAG retriever with temporal decay

    Fixes "temporal blindness" by:
    1. Extracting document publication dates
    2. Classifying temporal scope (market data vs evergreen)
    3. Applying exponential decay based on age and scope
    4. Flagging stale data with warnings
    """

    def __init__(self, base_retriever, verbose: bool = False):
        """
        Args:
            base_retriever: Existing retriever (TransformerRetriever)
            verbose: Print temporal decay details
        """
        self.retriever = base_retriever
        self.verbose = verbose
        self.date_extractor = DateExtractor()
        self.scope_classifier = TemporalScopeClassifier()

        # Precompute temporal metadata for all chunks
        self.temporal_metadata = {}
        self._precompute_metadata()

    def _precompute_metadata(self):
        """Precompute dates and scopes for all chunks"""
        if not hasattr(self.retriever, 'chunks'):
            return

        print("🕐 Precomputing temporal metadata...")

        for i, chunk in enumerate(self.retriever.chunks):
            chunk_id = self._get_chunk_id(chunk, i)

            # Extract date
            pub_date = self.date_extractor.extract_date(chunk)
            if not pub_date:
                pub_date = datetime.now()  # Assume current if unknown

            # Classify scope
            scope = self.scope_classifier.classify(chunk)

            self.temporal_metadata[chunk_id] = {
                'date': pub_date,
                'scope': scope,
                'profile': DECAY_PROFILES[scope]
            }

        print(f"✓ Computed metadata for {len(self.temporal_metadata)} chunks")

    def retrieve_with_decay(self, query: str, top_k: int = 5,
                           current_date: Optional[datetime] = None) -> List[Tuple[Dict, float]]:
        """
        Retrieve documents with temporal decay applied

        Args:
            query: Search query
            top_k: Number of results
            current_date: Current date (defaults to now)

        Returns:
            List of (chunk, decayed_score) tuples, sorted by decayed score
        """
        current_date = current_date or datetime.now()

        # Detect time-sensitive queries
        time_sensitive = self._is_time_sensitive_query(query)

        # Get more candidates than needed (will filter stale)
        candidates = self.retriever.search(query, top_k=top_k * 3)

        # Apply temporal decay
        scored_results = []
        for i, (chunk, score) in enumerate(candidates):
            chunk_id = self._get_chunk_id(chunk, i)

            if chunk_id in self.temporal_metadata:
                meta = self.temporal_metadata[chunk_id]
                pub_date = meta['date']
                profile = meta['profile']

                # Calculate age
                age_days = (current_date - pub_date).days

                # Apply exponential decay: score * 0.5^(age/half_life)
                if profile.half_life_days != float('inf'):
                    decay_factor = 0.5 ** (age_days / profile.half_life_days)
                else:
                    decay_factor = 1.0  # No decay for evergreen

                decayed_score = score * decay_factor

                # Flag stale data
                is_stale = age_days > profile.stale_threshold_days

                scored_results.append({
                    'chunk': chunk,
                    'original_score': score,
                    'decayed_score': decayed_score,
                    'age_days': age_days,
                    'scope': meta['scope'],
                    'is_stale': is_stale,
                    'pub_date': pub_date
                })

        # Sort by decayed score
        scored_results.sort(key=lambda x: x['decayed_score'], reverse=True)

        # Filter stale results for time-sensitive queries
        if time_sensitive:
            scored_results = [r for r in scored_results if not r['is_stale']]

        # Take top-k
        results = scored_results[:top_k]

        # Print decay info if verbose
        if self.verbose and results:
            self._print_decay_report(results, time_sensitive)

        # Return in expected format: [(chunk, score), ...]
        return [(r['chunk'], r['decayed_score']) for r in results]

    def _is_time_sensitive_query(self, query: str) -> bool:
        """Detect if query is time-sensitive"""
        time_keywords = [
            'current', 'latest', 'now', 'today', 'recent', 'new',
            'odds', 'price', 'rate', 'forecast', 'prediction',
            'will', 'when', 'how long', 'by when'
        ]

        query_lower = query.lower()
        return any(kw in query_lower for kw in time_keywords)

    def _get_chunk_id(self, chunk: Dict, index: int) -> str:
        """Generate unique ID for chunk"""
        return f"{chunk.get('metadata', {}).get('filename', 'unknown')}:{index}"

    def _print_decay_report(self, results: List[Dict], time_sensitive: bool):
        """Print temporal decay analysis"""
        print("\n" + "="*70)
        print(f"⏰ TEMPORAL DECAY ANALYSIS ({'time-sensitive' if time_sensitive else 'general'} query)")
        print("="*70)

        for i, r in enumerate(results, 1):
            age_str = f"{r['age_days']} days old"
            decay_pct = (1 - r['decayed_score'] / r['original_score']) * 100

            status = "⚠️  STALE" if r['is_stale'] else "✓ Fresh"

            print(f"\n{i}. {status} - {r['scope'].value}")
            print(f"   Age: {age_str} ({r['pub_date'].strftime('%Y-%m-%d')})")
            print(f"   Score: {r['original_score']:.3f} → {r['decayed_score']:.3f} ({decay_pct:.1f}% decay)")
            print(f"   Source: {r['chunk'].get('metadata', {}).get('filename', 'unknown')}")

        print("\n" + "="*70)


def main():
    """Test temporal decay system"""
    # Create test chunks with different ages
    test_chunks = [
        {
            'text': 'Betting odds: 3/1 that Starmer won\'t be PM by end of 2025',
            'metadata': {
                'filename': 'betting_analysis.md',
                'date': '2025-01-15'
            }
        },
        {
            'text': 'Current Betfair odds: 46/1 for Starmer resignation',
            'metadata': {
                'filename': 'current_markets.md',
                'date': '2025-11-07'
            }
        },
        {
            'text': 'Quicksort algorithm has O(n log n) average complexity',
            'metadata': {
                'filename': 'algorithms.md',
                'date': '2020-01-01'
            }
        },
        {
            'text': 'API documentation for user authentication endpoints',
            'metadata': {
                'filename': 'api_docs.md',
                'date': '2024-06-15'
            }
        },
    ]

    print("="*70)
    print("Temporal Decay Test")
    print("="*70)

    # Test date extraction
    print("\n📅 Date Extraction:")
    extractor = DateExtractor()
    for chunk in test_chunks:
        date = extractor.extract_date(chunk)
        print(f"  {chunk['metadata']['filename']}: {date}")

    # Test scope classification
    print("\n🔍 Scope Classification:")
    classifier = TemporalScopeClassifier()
    for chunk in test_chunks:
        scope = classifier.classify(chunk)
        print(f"  {chunk['metadata']['filename']}: {scope.value}")

    # Test decay calculation
    print("\n📉 Decay Calculation (current date: 2025-11-08):")
    current = datetime(2025, 11, 8)

    for chunk in test_chunks:
        date = extractor.extract_date(chunk)
        scope = classifier.classify(chunk)
        profile = DECAY_PROFILES[scope]

        age_days = (current - date).days
        if profile.half_life_days != float('inf'):
            decay = 0.5 ** (age_days / profile.half_life_days)
        else:
            decay = 1.0

        print(f"  {chunk['metadata']['filename']}:")
        print(f"    Age: {age_days} days | Decay: {decay:.3f} | Stale: {age_days > profile.stale_threshold_days}")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Query Difficulty Classifier

Classifies queries as simple/medium/hard to allocate appropriate compute budget.

Classification based on:
- Query complexity indicators (keywords, structure)
- Multi-step reasoning requirements
- Domain specificity
- Ambiguity and vagueness

Author: LAT5150DRVMIL AI Framework
Version: 1.0.0
"""

from enum import Enum
from typing import Tuple, List, Dict
import re


class DifficultyLevel(Enum):
    """Query difficulty levels"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    HARD = "hard"


class DifficultyClassifier:
    """
    Classify query difficulty for adaptive compute allocation

    Usage:
        classifier = DifficultyClassifier()
        difficulty, confidence = classifier.classify("Explain quantum entanglement")
        # Returns: (DifficultyLevel.HARD, 0.85)
    """

    def __init__(self):
        """Initialize difficulty classifier with detection patterns"""

        # Simple query indicators
        self.simple_patterns = {
            'questions': [
                r'\bwhat is\b',
                r'\bwho is\b',
                r'\bwhen\b',
                r'\bwhere\b',
                r'\bdefine\b',
                r'\blist\b',
                r'\bname\b'
            ],
            'commands': [
                r'\bshow\b',
                r'\bdisplay\b',
                r'\bprint\b',
                r'\bget\b'
            ]
        }

        # Medium query indicators
        self.medium_patterns = {
            'how_to': [
                r'\bhow to\b',
                r'\bhow do i\b',
                r'\bhow can i\b'
            ],
            'explanation': [
                r'\bexplain\b',
                r'\bdescribe\b',
                r'\bcompare\b',
                r'\bdifference between\b'
            ],
            'multi_part': [
                r'\band\b.*\band\b',  # Multiple "and"s
                r'\bor\b.*\bor\b'     # Multiple "or"s
            ]
        }

        # Hard query indicators
        self.hard_patterns = {
            'analysis': [
                r'\banalyze\b',
                r'\bevaluate\b',
                r'\bassess\b',
                r'\binvestigate\b',
                r'\bexplore\b'
            ],
            'design': [
                r'\bdesign\b',
                r'\barchitect\b',
                r'\boptimize\b',
                r'\bimplement\b',
                r'\bcreate system\b'
            ],
            'reasoning': [
                r'\bprove\b',
                r'\bderive\b',
                r'\bjustify\b',
                r'\bwhy\b.*\bwhy\b',  # Multiple "why"s
                r'\bcausality\b'
            ],
            'comprehensive': [
                r'\bcomprehensive\b',
                r'\bdetailed\b',
                r'\bin-depth\b',
                r'\bthorough\b'
            ],
            'multi_step': [
                r'\bstep[- ]by[- ]step\b',
                r'\bfirst.*then.*finally\b'
            ]
        }

        # Domain complexity keywords
        self.complex_domains = [
            'quantum', 'cryptography', 'distributed systems',
            'machine learning', 'optimization', 'algorithm design',
            'security architecture', 'performance tuning',
            'formal verification', 'theorem proving'
        ]

    def classify(self, query: str) -> Tuple[DifficultyLevel, float]:
        """
        Classify query difficulty

        Args:
            query: User query text

        Returns:
            Tuple of (DifficultyLevel, confidence_score)
            confidence_score: 0.0-1.0 confidence in classification
        """
        query_lower = query.lower()

        # Score for each difficulty level
        scores = {
            DifficultyLevel.SIMPLE: 0.0,
            DifficultyLevel.MEDIUM: 0.0,
            DifficultyLevel.HARD: 0.0
        }

        # 1. Pattern matching
        for pattern_list in self.simple_patterns.values():
            for pattern in pattern_list:
                if re.search(pattern, query_lower):
                    scores[DifficultyLevel.SIMPLE] += 1.0

        for pattern_list in self.medium_patterns.values():
            for pattern in pattern_list:
                if re.search(pattern, query_lower):
                    scores[DifficultyLevel.MEDIUM] += 1.0

        for pattern_list in self.hard_patterns.values():
            for pattern in pattern_list:
                if re.search(pattern, query_lower):
                    scores[DifficultyLevel.HARD] += 1.5  # Weight harder

        # 2. Domain complexity
        for domain in self.complex_domains:
            if domain in query_lower:
                scores[DifficultyLevel.HARD] += 2.0

        # 3. Query length (longer usually means more complex)
        word_count = len(query.split())
        if word_count <= 5:
            scores[DifficultyLevel.SIMPLE] += 0.5
        elif word_count <= 15:
            scores[DifficultyLevel.MEDIUM] += 0.5
        else:
            scores[DifficultyLevel.HARD] += 1.0

        # 4. Question marks (multiple questions = harder)
        question_count = query.count('?')
        if question_count > 1:
            scores[DifficultyLevel.HARD] += question_count * 0.5

        # 5. Technical terms and jargon
        technical_terms = self._count_technical_terms(query_lower)
        if technical_terms > 3:
            scores[DifficultyLevel.HARD] += technical_terms * 0.3

        # Determine difficulty by highest score
        if max(scores.values()) == 0:
            # No signals detected, default to medium
            return DifficultyLevel.MEDIUM, 0.5

        difficulty = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[difficulty] / total_score if total_score > 0 else 0.5

        return difficulty, min(confidence, 1.0)

    def _count_technical_terms(self, text: str) -> int:
        """Count technical terms in text"""
        technical_keywords = [
            'algorithm', 'optimization', 'architecture', 'framework',
            'implementation', 'performance', 'scalability', 'latency',
            'throughput', 'distributed', 'concurrent', 'parallel',
            'asynchronous', 'synchronous', 'protocol', 'interface'
        ]

        count = 0
        for term in technical_keywords:
            if term in text:
                count += 1

        return count

    def get_features(self, query: str) -> Dict:
        """
        Extract classification features for debugging

        Args:
            query: User query

        Returns:
            Dict of features used for classification
        """
        query_lower = query.lower()

        features = {
            'word_count': len(query.split()),
            'char_count': len(query),
            'question_marks': query.count('?'),
            'technical_terms': self._count_technical_terms(query_lower),
            'simple_patterns': 0,
            'medium_patterns': 0,
            'hard_patterns': 0,
            'complex_domain': False
        }

        # Count pattern matches
        for pattern_list in self.simple_patterns.values():
            for pattern in pattern_list:
                if re.search(pattern, query_lower):
                    features['simple_patterns'] += 1

        for pattern_list in self.medium_patterns.values():
            for pattern in pattern_list:
                if re.search(pattern, query_lower):
                    features['medium_patterns'] += 1

        for pattern_list in self.hard_patterns.values():
            for pattern in pattern_list:
                if re.search(pattern, query_lower):
                    features['hard_patterns'] += 1

        # Check complex domains
        for domain in self.complex_domains:
            if domain in query_lower:
                features['complex_domain'] = True
                break

        return features


if __name__ == "__main__":
    # Demo usage
    print("="*60)
    print("Difficulty Classifier Demo")
    print("="*60)

    classifier = DifficultyClassifier()

    test_queries = [
        "What is Python?",
        "How do I sort a list in Python?",
        "Explain the difference between TCP and UDP",
        "Design a distributed system for real-time analytics with fault tolerance",
        "Analyze the time complexity of quicksort and prove its average case",
        "List all files",
        "Implement a neural network from scratch and explain backpropagation",
        "When was Python created?",
        "Compare machine learning frameworks and recommend one for production"
    ]

    for query in test_queries:
        difficulty, confidence = classifier.classify(query)
        features = classifier.get_features(query)

        print(f"\nQuery: {query[:70]}...")
        print(f"  Difficulty: {difficulty.value.upper()}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Features: {features}")

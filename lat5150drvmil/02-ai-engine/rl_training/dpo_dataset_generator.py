#!/usr/bin/env python3
"""
DPO Dataset Generator

Generates preference pairs for DPO training from:
1. Human feedback (thumbs up/down, ratings)
2. A/B comparisons
3. Corrections
4. Simulated preferences from model outputs

Output format:
{
    "prompt": "User query",
    "chosen": "Preferred response",
    "rejected": "Non-preferred response"
}
"""

import os
import json
import sqlite3
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import random


@dataclass
class PreferencePair:
    """Single preference pair for DPO"""
    prompt: str
    chosen: str
    rejected: str
    source: str  # "human_feedback", "comparison", "correction", "simulated"
    confidence: float  # 0.0-1.0


class DPODatasetGenerator:
    """
    Generate DPO training dataset from various feedback sources
    """

    def __init__(
        self,
        feedback_db_path: str = "/home/user/LAT5150DRVMIL/data/feedback.db",
        output_path: str = "/home/user/LAT5150DRVMIL/02-ai-engine/training_data/dpo_preferences.json",
        min_confidence: float = 0.6
    ):
        self.feedback_db_path = feedback_db_path
        self.output_path = output_path
        self.min_confidence = min_confidence

        self.preference_pairs: List[PreferencePair] = []

    def connect_db(self) -> sqlite3.Connection:
        """Connect to feedback database"""
        if not Path(self.feedback_db_path).exists():
            print(f"⚠️  Feedback database not found: {self.feedback_db_path}")
            print("   Creating empty database...")
            self._create_empty_db()

        return sqlite3.connect(self.feedback_db_path)

    def _create_empty_db(self):
        """Create empty feedback database with schema"""
        os.makedirs(Path(self.feedback_db_path).parent, exist_ok=True)

        conn = sqlite3.connect(self.feedback_db_path)
        cursor = conn.cursor()

        # Create feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                query TEXT,
                response_a TEXT,
                response_b TEXT,
                feedback_type TEXT,
                feedback_value TEXT,
                timestamp REAL
            )
        ''')

        conn.commit()
        conn.close()

    def extract_from_thumbs(self) -> int:
        """
        Extract preferences from thumbs up/down feedback

        Thumbs up = chosen
        Thumbs down = rejected (need to generate alternative)
        """
        conn = self.connect_db()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT query, response_a, feedback_value
            FROM feedback
            WHERE feedback_type = 'thumbs'
        ''')

        added = 0
        for row in cursor.fetchall():
            query, response, feedback_json = row

            try:
                feedback = json.loads(feedback_json)
                thumbs = feedback.get('thumbs')

                if thumbs == 'up':
                    # This is a good response
                    # We need to generate a worse alternative
                    rejected = self._generate_worse_alternative(query, response)

                    if rejected:
                        self.preference_pairs.append(PreferencePair(
                            prompt=query,
                            chosen=response,
                            rejected=rejected,
                            source="human_feedback",
                            confidence=0.9
                        ))
                        added += 1

                elif thumbs == 'down':
                    # This is a bad response
                    # We need a better alternative
                    chosen = self._generate_better_alternative(query, response)

                    if chosen:
                        self.preference_pairs.append(PreferencePair(
                            prompt=query,
                            chosen=chosen,
                            rejected=response,
                            source="human_feedback",
                            confidence=0.8
                        ))
                        added += 1

            except json.JSONDecodeError:
                continue

        conn.close()
        print(f"✓ Extracted {added} preferences from thumbs feedback")
        return added

    def extract_from_comparisons(self) -> int:
        """
        Extract preferences from A/B comparisons

        Direct preference: chosen vs rejected
        """
        conn = self.connect_db()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT query, response_a, response_b, feedback_value
            FROM feedback
            WHERE feedback_type = 'comparison'
        ''')

        added = 0
        for row in cursor.fetchall():
            query, response_a, response_b, feedback_json = row

            try:
                feedback = json.loads(feedback_json)
                preferred = feedback.get('preferred')  # 'a' or 'b'

                if preferred == 'a':
                    chosen = response_a
                    rejected = response_b
                elif preferred == 'b':
                    chosen = response_b
                    rejected = response_a
                else:
                    continue

                self.preference_pairs.append(PreferencePair(
                    prompt=query,
                    chosen=chosen,
                    rejected=rejected,
                    source="comparison",
                    confidence=1.0  # Direct comparison is high confidence
                ))
                added += 1

            except json.JSONDecodeError:
                continue

        conn.close()
        print(f"✓ Extracted {added} preferences from comparisons")
        return added

    def extract_from_corrections(self) -> int:
        """
        Extract preferences from corrections

        Corrected response = chosen
        Original response = rejected
        """
        conn = self.connect_db()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT query, response_a, feedback_value
            FROM feedback
            WHERE feedback_type = 'correction'
        ''')

        added = 0
        for row in cursor.fetchall():
            query, original_response, feedback_json = row

            try:
                feedback = json.loads(feedback_json)
                corrected = feedback.get('corrected_response')

                if corrected and corrected != original_response:
                    self.preference_pairs.append(PreferencePair(
                        prompt=query,
                        chosen=corrected,
                        rejected=original_response,
                        source="correction",
                        confidence=0.95  # Corrections are high quality
                    ))
                    added += 1

            except json.JSONDecodeError:
                continue

        conn.close()
        print(f"✓ Extracted {added} preferences from corrections")
        return added

    def extract_from_ratings(self) -> int:
        """
        Extract preferences from ratings

        High rating (4-5) vs Low rating (1-2) for similar queries
        """
        conn = self.connect_db()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT query, response_a, feedback_value
            FROM feedback
            WHERE feedback_type = 'rating'
        ''')

        # Group by similar queries
        by_query = {}
        for row in cursor.fetchall():
            query, response, feedback_json = row

            try:
                feedback = json.loads(feedback_json)
                rating = feedback.get('rating')

                if rating:
                    if query not in by_query:
                        by_query[query] = []
                    by_query[query].append((response, rating))

            except json.JSONDecodeError:
                continue

        # Create preferences from high vs low ratings
        added = 0
        for query, responses in by_query.items():
            if len(responses) < 2:
                continue

            # Sort by rating
            responses.sort(key=lambda x: x[1], reverse=True)

            # Take best and worst
            best_response, best_rating = responses[0]
            worst_response, worst_rating = responses[-1]

            # Only if there's significant difference
            if best_rating >= 4 and worst_rating <= 2:
                self.preference_pairs.append(PreferencePair(
                    prompt=query,
                    chosen=best_response,
                    rejected=worst_response,
                    source="rating",
                    confidence=0.7
                ))
                added += 1

        conn.close()
        print(f"✓ Extracted {added} preferences from ratings")
        return added

    def _generate_worse_alternative(
        self,
        query: str,
        good_response: str
    ) -> Optional[str]:
        """
        Generate a worse alternative response

        Strategies:
        - Make it shorter and less detailed
        - Make it more generic
        - Introduce minor inaccuracies
        """
        # Simple strategy: truncate to first sentence
        sentences = good_response.split('.')
        if len(sentences) > 1:
            return sentences[0] + "."

        return None

    def _generate_better_alternative(
        self,
        query: str,
        bad_response: str
    ) -> Optional[str]:
        """
        Generate a better alternative response

        In practice, this would use a better model or retrieve from knowledge base
        For now, return None to skip (require real human data)
        """
        # TODO: Could use a better model here
        # For now, skip these
        return None

    def generate_simulated_pairs(self, num_pairs: int = 100) -> int:
        """
        Generate simulated preference pairs for bootstrapping

        Uses common patterns and synthetic data
        """
        templates = [
            {
                "prompt": "What is {topic}?",
                "chosen": "comprehensive explanation of {topic}",
                "rejected": "brief mention of {topic}"
            },
            {
                "prompt": "How do I {task}?",
                "chosen": "step-by-step guide for {task}",
                "rejected": "vague suggestion about {task}"
            },
            {
                "prompt": "Explain {concept}",
                "chosen": "detailed explanation with examples of {concept}",
                "rejected": "one-sentence description of {concept}"
            }
        ]

        topics = [
            "machine learning", "neural networks", "Python programming",
            "data structures", "algorithms", "databases", "web development",
            "cloud computing", "cybersecurity", "quantum computing"
        ]

        added = 0
        for _ in range(num_pairs):
            template = random.choice(templates)
            topic = random.choice(topics)

            prompt = template["prompt"].format(topic=topic, task=topic, concept=topic)
            chosen = template["chosen"].format(topic=topic, task=topic, concept=topic)
            rejected = template["rejected"].format(topic=topic, task=topic, concept=topic)

            self.preference_pairs.append(PreferencePair(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                source="simulated",
                confidence=0.5  # Lower confidence for simulated data
            ))
            added += 1

        print(f"✓ Generated {added} simulated preference pairs")
        return added

    def filter_by_confidence(self):
        """Filter preference pairs by confidence threshold"""
        before = len(self.preference_pairs)
        self.preference_pairs = [
            pair for pair in self.preference_pairs
            if pair.confidence >= self.min_confidence
        ]
        after = len(self.preference_pairs)

        print(f"✓ Filtered by confidence (>={self.min_confidence}): {before} → {after} pairs")

    def deduplicate(self):
        """Remove duplicate preference pairs"""
        seen = set()
        unique_pairs = []

        for pair in self.preference_pairs:
            key = (pair.prompt, pair.chosen, pair.rejected)
            if key not in seen:
                seen.add(key)
                unique_pairs.append(pair)

        before = len(self.preference_pairs)
        self.preference_pairs = unique_pairs
        after = len(self.preference_pairs)

        print(f"✓ Deduplicated: {before} → {after} pairs")

    def save_dataset(self):
        """Save preference pairs to JSON"""
        os.makedirs(Path(self.output_path).parent, exist_ok=True)

        # Convert to simple format for DPO trainer
        dataset = [
            {
                "prompt": pair.prompt,
                "chosen": pair.chosen,
                "rejected": pair.rejected,
                "metadata": {
                    "source": pair.source,
                    "confidence": pair.confidence
                }
            }
            for pair in self.preference_pairs
        ]

        with open(self.output_path, 'w') as f:
            json.dump(dataset, f, indent=2)

        print(f"\n✓ Saved {len(dataset)} preference pairs to: {self.output_path}")

    def generate(
        self,
        use_human_feedback: bool = True,
        use_simulated: bool = True,
        num_simulated: int = 100
    ):
        """Main generation pipeline"""
        print("\n" + "=" * 80)
        print("  DPO Dataset Generation")
        print("=" * 80)

        total_pairs = 0

        if use_human_feedback:
            print("\nExtracting from human feedback...")
            total_pairs += self.extract_from_thumbs()
            total_pairs += self.extract_from_comparisons()
            total_pairs += self.extract_from_corrections()
            total_pairs += self.extract_from_ratings()

        if use_simulated:
            print("\nGenerating simulated pairs...")
            total_pairs += self.generate_simulated_pairs(num_simulated)

        print(f"\nTotal pairs collected: {total_pairs}")

        # Post-processing
        print("\nPost-processing...")
        self.filter_by_confidence()
        self.deduplicate()

        # Statistics
        self._print_statistics()

        # Save
        self.save_dataset()

        print("\n" + "=" * 80)
        print("✅ Dataset generation complete!")
        print("=" * 80)

    def _print_statistics(self):
        """Print dataset statistics"""
        print("\nDataset Statistics:")
        print(f"  Total pairs: {len(self.preference_pairs)}")

        # By source
        by_source = {}
        for pair in self.preference_pairs:
            by_source[pair.source] = by_source.get(pair.source, 0) + 1

        print("\n  By source:")
        for source, count in sorted(by_source.items()):
            print(f"    {source}: {count}")

        # Confidence distribution
        confidences = [pair.confidence for pair in self.preference_pairs]
        if confidences:
            print(f"\n  Confidence: min={min(confidences):.2f}, "
                  f"max={max(confidences):.2f}, "
                  f"avg={sum(confidences)/len(confidences):.2f}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="DPO Dataset Generator")
    parser.add_argument("--feedback-db", default="/home/user/LAT5150DRVMIL/data/feedback.db",
                       help="Feedback database path")
    parser.add_argument("--output", default="/home/user/LAT5150DRVMIL/02-ai-engine/training_data/dpo_preferences.json",
                       help="Output JSON path")
    parser.add_argument("--no-simulated", action="store_true",
                       help="Don't generate simulated pairs")
    parser.add_argument("--num-simulated", type=int, default=100,
                       help="Number of simulated pairs")
    parser.add_argument("--min-confidence", type=float, default=0.6,
                       help="Minimum confidence threshold")

    args = parser.parse_args()

    generator = DPODatasetGenerator(
        feedback_db_path=args.feedback_db,
        output_path=args.output,
        min_confidence=args.min_confidence
    )

    generator.generate(
        use_simulated=not args.no_simulated,
        num_simulated=args.num_simulated
    )


if __name__ == "__main__":
    main()

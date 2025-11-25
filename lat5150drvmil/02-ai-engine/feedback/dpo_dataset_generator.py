#!/usr/bin/env python3
"""
DPO Dataset Generator

Generates training datasets for Direct Preference Optimization (DPO) from:
- User corrections (chosen vs rejected responses)
- Thumbs up/down feedback
- Rating comparisons (high-rated vs low-rated)
- Explicit preferences (A vs B choices)

DPO trains models to prefer chosen responses over rejected ones
without requiring reward models (unlike RLHF/PPO).

Author: LAT5150DRVMIL AI Framework
Version: 1.0.0
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from .hitl_feedback import HITLFeedbackCollector


class DPODatasetGenerator:
    """
    Generate DPO training datasets from HITL feedback

    Usage:
        generator = DPODatasetGenerator()
        dataset = generator.generate_dataset(min_pairs=100)
        generator.save_dataset(dataset, "dpo_training.json")
    """

    def __init__(self, feedback_db_path: Optional[str] = None):
        """
        Initialize DPO dataset generator

        Args:
            feedback_db_path: Path to HITL feedback database
        """
        self.collector = HITLFeedbackCollector(
            db_path=feedback_db_path or "~/.rag_index/hitl_feedback.db"
        )

    def generate_dataset(
        self,
        min_pairs: int = 100,
        include_ratings: bool = True,
        rating_threshold: int = 3
    ) -> List[Dict]:
        """
        Generate DPO training dataset

        Args:
            min_pairs: Minimum number of preference pairs
            include_ratings: Generate pairs from ratings (high vs low)
            rating_threshold: Ratings >= threshold are "chosen", < threshold are "rejected"

        Returns:
            List of DPO training examples:
            [
                {
                    "prompt": "How do I...",
                    "chosen": "Good response",
                    "rejected": "Bad response"
                },
                ...
            ]
        """
        dataset = []

        # 1. Get explicit preference pairs (corrections, A vs B choices)
        explicit_pairs = self.collector.export_dpo_pairs(min_examples=min_pairs)
        dataset.extend(explicit_pairs)

        print(f"Added {len(explicit_pairs)} explicit preference pairs")

        # 2. Generate pairs from ratings if enabled
        if include_ratings:
            rating_pairs = self._generate_rating_pairs(rating_threshold)
            dataset.extend(rating_pairs)
            print(f"Added {len(rating_pairs)} rating-based pairs")

        # 3. Deduplicate
        dataset = self._deduplicate_dataset(dataset)

        print(f"\nTotal DPO dataset size: {len(dataset)} pairs")
        return dataset

    def _generate_rating_pairs(self, threshold: int = 3) -> List[Dict]:
        """
        Generate preference pairs from ratings

        Strategy:
        - Group responses by query
        - Pair high-rated (>=threshold) with low-rated (<threshold) responses
        - Creates pairs: (query, high_rated_response, low_rated_response)
        """
        import sqlite3

        cursor = self.collector.conn.cursor()

        # Get all rated responses
        cursor.execute("""
            SELECT query, response, rating
            FROM feedback
            WHERE feedback_type = 'rating' AND rating IS NOT NULL
            ORDER BY query, rating DESC
        """)

        responses_by_query = {}
        for query, response, rating in cursor.fetchall():
            if query not in responses_by_query:
                responses_by_query[query] = {'high': [], 'low': []}

            if rating >= threshold:
                responses_by_query[query]['high'].append(response)
            else:
                responses_by_query[query]['low'].append(response)

        # Create pairs
        pairs = []
        for query, responses in responses_by_query.items():
            high = responses['high']
            low = responses['low']

            # Pair each high-rated with each low-rated (cartesian product)
            for chosen in high:
                for rejected in low:
                    pairs.append({
                        "prompt": query,
                        "chosen": chosen,
                        "rejected": rejected
                    })

        return pairs

    def _deduplicate_dataset(self, dataset: List[Dict]) -> List[Dict]:
        """Remove duplicate pairs"""
        seen = set()
        deduplicated = []

        for item in dataset:
            # Create hash of (prompt, chosen, rejected)
            key = (item['prompt'], item['chosen'], item['rejected'])
            if key not in seen:
                seen.add(key)
                deduplicated.append(item)

        removed = len(dataset) - len(deduplicated)
        if removed > 0:
            print(f"Removed {removed} duplicate pairs")

        return deduplicated

    def save_dataset(self, dataset: List[Dict], output_path: str):
        """
        Save DPO dataset to JSON file

        Args:
            dataset: DPO dataset from generate_dataset()
            output_path: Path to save JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)

        print(f"✓ Saved DPO dataset: {output_path} ({len(dataset)} pairs)")

    def validate_dataset(self, dataset: List[Dict]) -> Dict:
        """
        Validate DPO dataset quality

        Args:
            dataset: DPO dataset to validate

        Returns:
            Validation statistics
        """
        stats = {
            'total_pairs': len(dataset),
            'unique_prompts': len(set(item['prompt'] for item in dataset)),
            'avg_chosen_length': 0,
            'avg_rejected_length': 0,
            'empty_chosen': 0,
            'empty_rejected': 0,
            'identical_pairs': 0
        }

        if not dataset:
            return stats

        chosen_lengths = []
        rejected_lengths = []

        for item in dataset:
            # Check for empty responses
            if not item['chosen'].strip():
                stats['empty_chosen'] += 1
            if not item['rejected'].strip():
                stats['empty_rejected'] += 1

            # Check for identical chosen/rejected
            if item['chosen'] == item['rejected']:
                stats['identical_pairs'] += 1

            # Lengths
            chosen_lengths.append(len(item['chosen']))
            rejected_lengths.append(len(item['rejected']))

        stats['avg_chosen_length'] = sum(chosen_lengths) / len(chosen_lengths)
        stats['avg_rejected_length'] = sum(rejected_lengths) / len(rejected_lengths)

        return stats

    def close(self):
        """Close feedback collector"""
        self.collector.close()


if __name__ == "__main__":
    # Demo usage
    print("="*60)
    print("DPO Dataset Generator Demo")
    print("="*60)

    generator = DPODatasetGenerator()

    # Generate dataset
    dataset = generator.generate_dataset(min_pairs=10, include_ratings=True)

    if dataset:
        # Validate
        validation = generator.validate_dataset(dataset)
        print("\nDataset Validation:")
        for key, value in validation.items():
            print(f"  {key}: {value}")

        # Save
        generator.save_dataset(dataset, "02-ai-engine/feedback/dpo_demo_dataset.json")

        # Show sample
        print("\nSample DPO pair:")
        sample = dataset[0]
        print(f"Prompt: {sample['prompt'][:80]}...")
        print(f"Chosen: {sample['chosen'][:80]}...")
        print(f"Rejected: {sample['rejected'][:80]}...")

    generator.close()

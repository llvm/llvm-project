#!/usr/bin/env python3
"""
User Feedback Collection System
Collects 1-10 ratings for RAG query results

This data is used for:
- TRL reward model training
- DPO (Direct Preference Optimization)
- System improvement metrics
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class FeedbackCollector:
    """Collect and manage user feedback on query results"""

    def __init__(self, feedback_file='rag_system/user_feedback.jsonl'):
        """
        Initialize feedback collector

        Args:
            feedback_file: Path to JSONL file for storing feedback
        """
        self.feedback_file = feedback_file
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def collect_rating(
        self,
        query: str,
        results: List[Dict],
        rating: Optional[int] = None,
        comment: Optional[str] = None,
        auto_prompt: bool = True
    ) -> Dict:
        """
        Collect user rating for query results

        Args:
            query: The user's query
            results: List of (chunk, score) results
            rating: 1-10 rating (if None, prompts user)
            comment: Optional text comment
            auto_prompt: Automatically prompt for rating

        Returns:
            Feedback dict
        """
        # Auto-prompt for rating if not provided
        if rating is None and auto_prompt:
            rating, auto_comment = self._prompt_for_rating()
            # Use auto-comment if no manual comment provided
            if auto_comment and not comment:
                comment = auto_comment

        # Create feedback entry
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'query': query,
            'rating': rating,
            'comment': comment or '',
            'results': [
                {
                    'filepath': result[0]['metadata']['filepath'],
                    'score': float(result[1]),
                    'preview': result[0]['text'][:200]
                }
                for result in results[:3]  # Top 3 results
            ],
            'result_count': len(results)
        }

        # Save to file
        self._save_feedback(feedback)

        return feedback

    def _prompt_for_rating(self, quick_mode: bool = True) -> int:
        """
        Prompt user for rating with quick feedback shortcuts

        Args:
            quick_mode: Show quick feedback shortcuts
        """
        print()
        print("─" * 60)
        print("📊 How would you rate these results?")
        print()

        if quick_mode:
            print("  Quick feedback (auto-rated):")
            print("    ✓ or 'perfect'       → 10 (exactly what I needed)")
            print("    ✅ or 'error free'   → 10 (code works perfectly)")
            print("    'compiles'           → 9  (code compiles)")
            print("    👍 or 'good'         → 8  (helpful and relevant)")
            print("    'almost'             → 7  (almost there)")
            print("    'partial'            → 6  (partially helpful)")
            print("    'incomplete'         → 5  (missing info)")
            print("    'unclear'            → 4  (confusing)")
            print("    👎 or 'bad'          → 3  (not relevant)")
            print("    ❌ or 'fails'        → 2  (code fails)")
            print("    'broken' or 'wrong'  → 1  (wrong info)")
            print()
            print("  Or enter 1-10, or 's' to skip")
            print("─" * 60)
        else:
            print("  1-3:   Poor (not relevant, wrong information)")
            print("  4-6:   Fair (somewhat relevant, but missing key info)")
            print("  7-8:   Good (relevant and helpful)")
            print("  9-10:  Excellent (exactly what I needed)")
            print()
            print("  0 or 's': Skip rating")
            print("─" * 60)

        # Quick feedback shortcuts
        shortcuts = {
            # Perfect/Excellent
            '✓': (10, "Perfect - exactly what I needed"),
            'perfect': (10, "Perfect - exactly what I needed"),
            'excellent': (10, "Excellent results"),

            # Good/Helpful
            '👍': (8, "Good - helpful and relevant"),
            'good': (8, "Good - helpful and relevant"),
            'helpful': (8, "Helpful information"),

            # Code works perfectly
            '✅': (10, "Code works perfectly - error free"),
            'compiles': (9, "Code compiles successfully"),
            'works': (9, "Code/solution works"),
            'error free': (10, "Code runs without errors"),
            'errorless': (10, "No errors encountered"),
            'clean': (10, "Clean execution, no issues"),

            # Code has issues
            '❌': (2, "Code fails/doesn't compile"),
            'fails': (2, "Code or solution fails"),
            'error': (2, "Runtime or compile error"),
            'broken': (1, "Wrong or broken code/information"),
            'crash': (1, "Code crashes or fails badly"),

            # Bad/Not helpful
            '👎': (3, "Bad - not relevant"),
            'bad': (3, "Not relevant or helpful"),
            'irrelevant': (3, "Information not relevant"),
            'wrong': (1, "Wrong information"),
            '🐛': (1, "Has bugs or issues"),

            # Partially helpful
            'unclear': (4, "Information unclear"),
            'incomplete': (5, "Information incomplete"),
            'partial': (6, "Partially helpful"),
            'almost': (7, "Almost what I needed"),
        }

        while True:
            try:
                user_input = input("Rating: ").strip().lower()

                if user_input in ['0', 's', 'skip', '']:
                    return None, None

                # Check shortcuts
                if user_input in shortcuts:
                    rating, comment = shortcuts[user_input]
                    print(f"   → Rated {rating}/10: {comment}")
                    return rating, comment

                # Try numeric rating
                rating = int(user_input)

                if 1 <= rating <= 10:
                    return rating, None
                else:
                    print("❌ Please enter a number between 1 and 10")

            except ValueError:
                print("❌ Invalid input. Use a number (1-10), shortcut, or 's' to skip")
            except KeyboardInterrupt:
                print("\nSkipping rating...")
                return None, None

    def _save_feedback(self, feedback: Dict):
        """Append feedback to JSONL file"""
        with open(self.feedback_file, 'a') as f:
            f.write(json.dumps(feedback) + '\n')

    def get_statistics(self) -> Dict:
        """Get feedback statistics"""
        if not Path(self.feedback_file).exists():
            return {
                'total_ratings': 0,
                'average_rating': 0.0,
                'rating_distribution': {}
            }

        ratings = []
        rating_dist = {i: 0 for i in range(1, 11)}

        with open(self.feedback_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get('rating') is not None:
                        rating = entry['rating']
                        ratings.append(rating)
                        rating_dist[rating] = rating_dist.get(rating, 0) + 1
                except json.JSONDecodeError:
                    continue

        return {
            'total_ratings': len(ratings),
            'average_rating': sum(ratings) / len(ratings) if ratings else 0.0,
            'rating_distribution': rating_dist,
            'ratings': ratings
        }

    def print_statistics(self):
        """Print feedback statistics"""
        stats = self.get_statistics()

        print("\n" + "=" * 60)
        print("📊 User Feedback Statistics")
        print("=" * 60)
        print()

        if stats['total_ratings'] == 0:
            print("No ratings collected yet.")
            print()
            return

        print(f"Total Ratings: {stats['total_ratings']}")
        print(f"Average Rating: {stats['average_rating']:.2f}/10")
        print()

        print("Rating Distribution:")
        for rating in range(10, 0, -1):
            count = stats['rating_distribution'].get(rating, 0)
            bar_length = int(count / stats['total_ratings'] * 40) if stats['total_ratings'] > 0 else 0
            bar = "█" * bar_length
            print(f"  {rating:2d} | {bar} {count}")

        print()
        print("Quality Breakdown:")
        poor = sum(stats['rating_distribution'].get(i, 0) for i in range(1, 4))
        fair = sum(stats['rating_distribution'].get(i, 0) for i in range(4, 7))
        good = sum(stats['rating_distribution'].get(i, 0) for i in range(7, 9))
        excellent = sum(stats['rating_distribution'].get(i, 0) for i in range(9, 11))

        total = stats['total_ratings']
        if total > 0:
            print(f"  Poor (1-3):      {poor:3d} ({poor/total*100:5.1f}%)")
            print(f"  Fair (4-6):      {fair:3d} ({fair/total*100:5.1f}%)")
            print(f"  Good (7-8):      {good:3d} ({good/total*100:5.1f}%)")
            print(f"  Excellent (9-10):{excellent:3d} ({excellent/total*100:5.1f}%)")

        print()
        print("=" * 60)
        print()

    def export_for_training(self, output_file='rag_system/feedback_training_data.json'):
        """
        Export feedback data for reward model training

        Creates preference pairs for TRL/DPO training:
        - High-rated (7-10) vs low-rated (1-6) results
        """
        if not Path(self.feedback_file).exists():
            print("No feedback data available.")
            return

        high_quality = []  # Rating >= 7
        low_quality = []   # Rating <= 6

        with open(self.feedback_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    rating = entry.get('rating')

                    if rating is None:
                        continue

                    if rating >= 7:
                        high_quality.append(entry)
                    else:
                        low_quality.append(entry)

                except json.JSONDecodeError:
                    continue

        # Create preference pairs
        preference_pairs = []

        for high in high_quality:
            # Find similar query with low rating (if any)
            for low in low_quality:
                # Simple similarity: same first word
                high_first = high['query'].split()[0].lower() if high['query'] else ''
                low_first = low['query'].split()[0].lower() if low['query'] else ''

                if high_first and high_first == low_first:
                    preference_pairs.append({
                        'query': high['query'],
                        'chosen': high['results'][0] if high['results'] else None,
                        'rejected': low['results'][0] if low['results'] else None,
                        'chosen_rating': high['rating'],
                        'rejected_rating': low['rating']
                    })

        # Export
        training_data = {
            'high_quality_samples': high_quality,
            'low_quality_samples': low_quality,
            'preference_pairs': preference_pairs,
            'stats': {
                'total_feedback': len(high_quality) + len(low_quality),
                'high_quality': len(high_quality),
                'low_quality': len(low_quality),
                'preference_pairs': len(preference_pairs)
            }
        }

        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2)

        print(f"✓ Exported training data to {output_file}")
        print(f"  High-quality samples: {len(high_quality)}")
        print(f"  Low-quality samples: {len(low_quality)}")
        print(f"  Preference pairs: {len(preference_pairs)}")
        print()


def main():
    """Test feedback collector"""
    collector = FeedbackCollector()

    # Show current statistics
    collector.print_statistics()

    # Export for training
    try:
        collector.export_for_training()
    except Exception as e:
        print(f"Could not export training data: {e}")


if __name__ == '__main__':
    main()

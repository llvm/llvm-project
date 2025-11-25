#!/usr/bin/env python3
"""
PEFT Fine-Tuning Data Preparation
Generates training pairs from LAT5150DRVMIL documentation

Uses datatrove for efficient data processing pipelines

This creates:
- Positive pairs (query, relevant_document)
- Hard negatives (query, similar_but_irrelevant_document)
- Domain-specific vocabulary emphasis
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

# Try to use datatrove for efficient data processing
try:
    from datatrove.pipeline.readers import JsonlReader
    from datatrove.pipeline.writers import JsonlWriter
    DATATROVE_AVAILABLE = True
except ImportError:
    DATATROVE_AVAILABLE = False
    # Fallback to standard processing


class PEFTDataGenerator:
    """Generate fine-tuning data for embedding model"""

    def __init__(self, chunks_path='rag_system/processed_docs.json'):
        """Load document chunks"""
        with open(chunks_path, 'r') as f:
            data = json.load(f)

        self.chunks = data['chunks']
        self.stats = data['stats']

        # Extract domain-specific terms
        self.domain_terms = self._extract_domain_terms()

    def _extract_domain_terms(self) -> List[str]:
        """Extract LAT5150DRVMIL-specific terminology"""
        terms = set()

        # Known domain terms
        known_terms = [
            'DSMIL', 'NPU', 'APT41', 'VAULT7', 'TPM', 'ZFS',
            'Meteor Lake', 'Intel MTL', 'kernel', 'unified platform',
            'military', 'security hardening', 'activation', 'bootloader'
        ]
        terms.update(known_terms)

        # Extract from file paths (categories)
        for chunk in self.chunks:
            filepath = chunk['metadata']['filepath']
            # Extract capitalized terms from filenames
            filename = Path(filepath).stem
            words = re.findall(r'[A-Z][A-Z0-9_]+', filename)
            terms.update(words)

        return sorted(list(terms))

    def generate_query_document_pairs(self, num_pairs: int = 1000) -> List[Tuple[str, str, int]]:
        """
        Generate (query, document, label) pairs

        Label: 1 = relevant, 0 = irrelevant
        """
        pairs = []

        # Strategy 1: Extract questions from documents (30%)
        pairs.extend(self._generate_from_documents(int(num_pairs * 0.3)))

        # Strategy 2: Synthetic queries from domain terms (40%)
        pairs.extend(self._generate_synthetic_queries(int(num_pairs * 0.4)))

        # Strategy 3: Hard negative mining (30%)
        pairs.extend(self._generate_hard_negatives(int(num_pairs * 0.3)))

        # Shuffle
        random.shuffle(pairs)

        return pairs

    def _generate_from_documents(self, num_pairs: int) -> List[Tuple[str, str, int]]:
        """Extract natural query-document pairs from text"""
        pairs = []

        # Patterns that indicate questions/topics
        question_patterns = [
            r'(?:What|How|Why|When|Where)\s+[^?]+\?',
            r'(?:To|For)\s+\w+\s+(?:the|a)\s+\w+',
            r'(?:Understanding|Implementing|Enabling|Configuring)\s+\w+'
        ]

        for chunk in random.sample(self.chunks, min(num_pairs * 2, len(self.chunks))):
            text = chunk['text']

            # Try to extract questions
            for pattern in question_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    query = matches[0].strip()
                    # Positive pair
                    pairs.append((query, text, 1))

                    # Negative pair (random chunk)
                    neg_chunk = random.choice(self.chunks)
                    pairs.append((query, neg_chunk['text'], 0))

                    if len(pairs) >= num_pairs * 2:
                        break

            if len(pairs) >= num_pairs * 2:
                break

        return pairs[:num_pairs * 2]  # num_pairs positive + num_pairs negative

    def _generate_synthetic_queries(self, num_pairs: int) -> List[Tuple[str, str, int]]:
        """Generate synthetic queries using domain terms"""
        pairs = []

        query_templates = [
            "What is {}?",
            "How to enable {}?",
            "How to configure {}?",
            "{} activation procedure",
            "{} security features",
            "Implementing {}",
            "Understanding {}",
            "{} in LAT5150DRVMIL",
            "Troubleshooting {}",
            "{} best practices"
        ]

        # Build term -> chunks index
        term_index = defaultdict(list)
        for chunk in self.chunks:
            text_lower = chunk['text'].lower()
            for term in self.domain_terms:
                if term.lower() in text_lower:
                    term_index[term].append(chunk)

        # Generate queries
        for term in self.domain_terms:
            if term not in term_index or len(term_index[term]) < 2:
                continue

            template = random.choice(query_templates)
            query = template.format(term)

            # Positive pair - chunk containing the term
            pos_chunk = random.choice(term_index[term])
            pairs.append((query, pos_chunk['text'], 1))

            # Negative pair - chunk NOT containing the term
            neg_chunks = [c for c in self.chunks if term.lower() not in c['text'].lower()]
            if neg_chunks:
                neg_chunk = random.choice(neg_chunks)
                pairs.append((query, neg_chunk['text'], 0))

            if len(pairs) >= num_pairs * 2:
                break

        return pairs[:num_pairs * 2]

    def _generate_hard_negatives(self, num_pairs: int) -> List[Tuple[str, str, int]]:
        """Generate hard negative pairs (similar but not relevant)"""
        pairs = []

        # Group chunks by category
        category_groups = defaultdict(list)
        for chunk in self.chunks:
            # Extract category from path (first directory)
            filepath = chunk['metadata']['filepath']
            category = filepath.split('/')[0] if '/' in filepath else 'root'
            category_groups[category].append(chunk)

        # Generate pairs across categories (hard negatives)
        categories = list(category_groups.keys())

        for i in range(num_pairs):
            # Pick two different categories
            if len(categories) < 2:
                break

            cat1, cat2 = random.sample(categories, 2)

            # Pick chunks from each category
            chunk1 = random.choice(category_groups[cat1])
            chunk2 = random.choice(category_groups[cat2])

            # Create query from chunk1, pair with chunk2 (negative)
            # Extract first sentence as query
            sentences = re.split(r'[.!?]+', chunk1['text'])
            query = sentences[0].strip() if sentences else chunk1['text'][:100]

            # Positive pair
            pairs.append((query, chunk1['text'], 1))

            # Hard negative pair (different category, potentially similar)
            pairs.append((query, chunk2['text'], 0))

        return pairs

    def save_training_data(self, output_path='rag_system/peft_training_data.json'):
        """Generate and save training data"""
        print("Generating PEFT fine-tuning dataset...")
        print(f"Domain terms: {len(self.domain_terms)}")
        print()

        # Generate pairs
        pairs = self.generate_query_document_pairs(num_pairs=1000)

        # Split into train/val
        split_idx = int(len(pairs) * 0.9)
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]

        # Format for training
        train_data = [
            {
                'query': query,
                'document': doc,
                'label': label
            }
            for query, doc, label in train_pairs
        ]

        val_data = [
            {
                'query': query,
                'document': doc,
                'label': label
            }
            for query, doc, label in val_pairs
        ]

        # Save
        data = {
            'train': train_data,
            'validation': val_data,
            'domain_terms': self.domain_terms,
            'stats': {
                'train_size': len(train_data),
                'val_size': len(val_data),
                'positive_ratio': sum(1 for d in train_data if d['label'] == 1) / len(train_data),
                'domain_terms': len(self.domain_terms)
            }
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Saved training data to {output_path}")
        print(f"  Train samples: {len(train_data)}")
        print(f"  Validation samples: {len(val_data)}")
        print(f"  Positive ratio: {data['stats']['positive_ratio']:.1%}")
        print()

        return data


def main():
    print("=" * 70)
    print("PEFT Fine-Tuning Data Preparation")
    print("=" * 70)
    print()

    generator = PEFTDataGenerator()
    data = generator.save_training_data()

    print("=" * 70)
    print("Sample training pairs:")
    print("=" * 70)

    # Show samples
    for i, sample in enumerate(data['train'][:3], 1):
        print(f"\nSample {i}:")
        print(f"  Query: {sample['query'][:80]}...")
        print(f"  Doc: {sample['document'][:80]}...")
        print(f"  Label: {'Relevant' if sample['label'] == 1 else 'Irrelevant'}")

    print()
    print("✓ Data preparation complete!")
    print()
    print("Next step: Run peft_finetune.py to train the model")
    print()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Open R1 Reasoning Layer for RAG System
Integrates OpenR1-Distill-7B for advanced reasoning over retrieved documents

This module adds chain-of-thought reasoning capabilities to the RAG system,
enabling step-by-step logical inference for complex queries.

Features:
- Automatic model download with INT4 quantization (saves disk space)
- CPU-optimized inference with Intel IPEX
- Chain-of-thought reasoning over RAG results
- Explanation generation for answers
"""

import os
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from optimum.quanto import quantize, freeze, qint4
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("❌ Missing dependencies!")
    print("Install with: pip install transformers optimum-quanto")
    TRANSFORMERS_AVAILABLE = False
    import sys
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "openr1-community/OpenR1-Distill-7B"
CACHE_DIR = Path("rag_system/models/openr1")
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7

# Intel CPU optimizations
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
    logger.info("Intel Extension for PyTorch available - enabling optimizations")
except ImportError:
    IPEX_AVAILABLE = False
    logger.warning("IPEX not available - running without Intel optimizations")


class OpenR1Reasoner:
    """
    Open R1 reasoning model for advanced RAG queries

    Provides chain-of-thought reasoning over retrieved documents,
    generating step-by-step explanations and verified answers.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        quantize_model: bool = True,
        use_ipex: bool = True
    ):
        """
        Initialize Open R1 reasoning model

        Args:
            model_name: HuggingFace model identifier
            quantize_model: Apply INT4 quantization (recommended for CPU)
            use_ipex: Enable Intel CPU optimizations
        """
        logger.info(f"Initializing Open R1 Reasoner: {model_name}")

        self.model_name = model_name
        self.device = "cpu"  # CPU-only for now
        self.quantized = quantize_model
        self.ipex_enabled = use_ipex and IPEX_AVAILABLE

        # Create cache directory
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Load model and tokenizer
        self._load_model()

        logger.info(f"Open R1 Reasoner ready (quantized={self.quantized}, ipex={self.ipex_enabled})")

    def _load_model(self):
        """Load and optimize Open R1 model"""
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=CACHE_DIR,
            trust_remote_code=True
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading model (this may take a few minutes on first run)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.float32,  # CPU uses FP32
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        # Move to CPU
        self.model = self.model.to(self.device)
        self.model.eval()

        # Apply quantization
        if self.quantized:
            logger.info("Applying INT4 quantization for efficient CPU inference...")
            quantize(self.model, weights=qint4)
            freeze(self.model)
            logger.info("✓ Model quantized to INT4 (~4x smaller, 2-3x faster)")

        # Apply Intel optimizations
        if self.ipex_enabled:
            logger.info("Applying Intel IPEX optimizations...")
            self.model = ipex.optimize(self.model)
            logger.info("✓ IPEX optimizations applied")

    def _create_reasoning_prompt(
        self,
        query: str,
        retrieved_docs: List[Dict],
        max_context: int = 2000
    ) -> str:
        """
        Create a reasoning prompt from query and retrieved documents

        Args:
            query: User's question
            retrieved_docs: List of retrieved document chunks with scores
            max_context: Maximum context length in characters

        Returns:
            Formatted prompt for Open R1
        """
        # Build context from retrieved documents
        context_parts = []
        context_length = 0

        for i, (doc, score) in enumerate(retrieved_docs[:3], 1):
            chunk_text = doc.get('chunk', '')
            source = doc.get('source', 'Unknown')

            chunk_preview = chunk_text[:max_context - context_length]
            context_parts.append(
                f"[Document {i}] (Relevance: {score:.2%})\n"
                f"Source: {source}\n"
                f"{chunk_preview}\n"
            )

            context_length += len(chunk_preview)
            if context_length >= max_context:
                break

        context = "\n".join(context_parts)

        # Create reasoning prompt (Open R1 format)
        prompt = f"""<think>
Given the following context from a knowledge base, provide a reasoned answer to the question.

Show your reasoning step-by-step, then provide a clear final answer.

Context:
{context}

Question: {query}

Let me think through this step by step:
</think>

<answer>"""

        return prompt

    def reason(
        self,
        query: str,
        retrieved_docs: List[Tuple[Dict, float]],
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE
    ) -> Dict:
        """
        Generate reasoned answer using Open R1

        Args:
            query: User's question
            retrieved_docs: Retrieved documents from RAG system
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more focused)

        Returns:
            Dict with reasoning trace and final answer
        """
        logger.info(f"Reasoning over {len(retrieved_docs)} retrieved documents...")

        # Create reasoning prompt
        prompt = self._create_reasoning_prompt(query, retrieved_docs)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Generate reasoning
        logger.info("Generating reasoned answer...")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract reasoning and answer
        reasoning_trace = ""
        final_answer = ""

        if "<think>" in full_response and "</think>" in full_response:
            think_start = full_response.index("<think>") + 7
            think_end = full_response.index("</think>")
            reasoning_trace = full_response[think_start:think_end].strip()

        if "<answer>" in full_response:
            answer_start = full_response.index("<answer>") + 8
            if "</answer>" in full_response:
                answer_end = full_response.index("</answer>")
                final_answer = full_response[answer_start:answer_end].strip()
            else:
                final_answer = full_response[answer_start:].strip()

        # Fallback if parsing fails
        if not final_answer:
            final_answer = full_response[len(prompt):].strip()

        return {
            'query': query,
            'reasoning': reasoning_trace,
            'answer': final_answer,
            'full_response': full_response,
            'num_docs': len(retrieved_docs)
        }

    def get_model_info(self) -> Dict:
        """Get information about loaded model"""
        import sys

        # Estimate model size
        param_count = sum(p.numel() for p in self.model.parameters())
        param_size_mb = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        ) / (1024 ** 2)

        return {
            'model_name': self.model_name,
            'device': self.device,
            'quantized': self.quantized,
            'ipex_enabled': self.ipex_enabled,
            'parameters': f"{param_count:,}",
            'model_size_mb': f"{param_size_mb:.1f} MB",
            'python_version': sys.version.split()[0],
            'torch_version': torch.__version__
        }


# Convenience function for quick testing
def test_reasoning(query: str = "What is the DSMIL AI system?"):
    """
    Test Open R1 reasoning with a sample query

    This function loads the transformer RAG system and uses Open R1
    to generate a reasoned answer.
    """
    print("=" * 70)
    print("Open R1 Reasoning Test")
    print("=" * 70)
    print()

    # Load RAG system
    print("Loading RAG system...")
    from transformer_upgrade import TransformerRetriever
    import json

    with open('rag_system/processed_docs.json', 'r') as f:
        data = json.load(f)

    retriever = TransformerRetriever(data['chunks'])

    # Search for documents
    print(f"Query: {query}")
    print()
    results = retriever.search(query, top_k=3)

    print(f"Retrieved {len(results)} documents")
    print()

    # Initialize Open R1
    print("Initializing Open R1 Reasoner...")
    reasoner = OpenR1Reasoner(quantize_model=True, use_ipex=True)

    # Get model info
    info = reasoner.get_model_info()
    print()
    print("Model Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()

    # Generate reasoned answer
    print("Generating reasoned answer...")
    print("-" * 70)
    result = reasoner.reason(query, results)

    print()
    print("REASONING TRACE:")
    print(result['reasoning'])
    print()
    print("FINAL ANSWER:")
    print(result['answer'])
    print()
    print("=" * 70)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Open R1 Reasoning Layer')
    parser.add_argument(
        '--query',
        type=str,
        help='Query to test reasoning with'
    )
    parser.add_argument(
        '--no-quantize',
        action='store_true',
        help='Disable INT4 quantization'
    )
    parser.add_argument(
        '--no-ipex',
        action='store_true',
        help='Disable Intel IPEX optimizations'
    )

    args = parser.parse_args()

    if args.query:
        # Custom query
        reasoner = OpenR1Reasoner(
            quantize_model=not args.no_quantize,
            use_ipex=not args.no_ipex
        )

        from transformer_upgrade import TransformerRetriever
        import json

        with open('rag_system/processed_docs.json', 'r') as f:
            data = json.load(f)

        retriever = TransformerRetriever(data['chunks'])
        results = retriever.search(args.query, top_k=3)

        result = reasoner.reason(args.query, results)

        print("REASONING:")
        print(result['reasoning'])
        print()
        print("ANSWER:")
        print(result['answer'])
    else:
        # Run default test
        test_reasoning()

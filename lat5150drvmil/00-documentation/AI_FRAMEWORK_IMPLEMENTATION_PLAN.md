# AI Framework Full Implementation Plan
## Hardware-Optimized Experimental Research Deployment

**Target Platform:** Dell Latitude 5450 MIL-SPEC
**Hardware Accelerators:** Intel NPU (49.4 TOPS), Arc GPU (16 TFLOPS), NCS2 (3x), AVX-512
**Timeline:** 18-24 months
**Focus:** Research papers + experimental methods + hardware optimization

**Date:** 2025-11-08
**Version:** 1.0

---

## Hardware Capabilities Analysis

### Available Accelerators

| Hardware | Capabilities | Optimal Use Cases | Limitations |
|----------|-------------|-------------------|-------------|
| **Intel NPU** | 49.4 TOPS INT8, 34 TOPS INT4 | Inference (quantized models), continuous tasks | Training limited, INT8/INT4 only |
| **Intel Arc GPU** | 16 TFLOPS FP16, 8 Xe-cores | Training (small models), RAG embeddings | 12GB VRAM, not optimal for large models |
| **Intel NCS2** | 1 TOPS per stick (3x total) | Parallel inference, edge deployment | USB bottleneck, limited VRAM |
| **AVX-512** | 2x FP32 throughput vs AVX2 | CPU-based inference, preprocessing | Power consumption, heat |
| **Intel GNA 3.5** | Audio/voice processing | Voice UI, audio embeddings | Audio-only |

### Hardware Constraints

**Critical Limitations:**
- ❌ Cannot train large models (7B+) on Arc GPU (insufficient VRAM)
- ❌ Cannot do full PPO training locally (requires multi-GPU cluster)
- ❌ NPU limited to INT8/INT4 quantized inference only

**Viable Strategies:**
- ✅ Train small models (125M-1.3B params) on Arc GPU
- ✅ Use NPU for continuous inference (RAG retrieval, routing)
- ✅ Hybrid training: Arc GPU for gradients, NPU for inference
- ✅ Cloud GPUs for RL training, deploy to local hardware
- ✅ LoRA/PEFT for parameter-efficient fine-tuning

---

## PHASE 1: DPO Training Pipeline (Weeks 1-6)

### Goal: Enable Self-Improvement via Direct Preference Optimization

**Why DPO First:**
- Simpler than PPO (no reward model, no RL loop)
- Can train on Arc GPU (small models)
- Uses existing `dpo_dataset_generator.py`
- Quick wins for agent improvement

### Research Papers

1. **"Direct Preference Optimization"** (Rafailov et al., 2023)
   - Main DPO paper
   - Binary cross-entropy loss on preference pairs
   - No separate reward model
   - Stability advantages over PPO

2. **"KTO: Kahneman-Tversky Optimization"** (Ethayarajh et al., 2024)
   - Even simpler than DPO
   - Uses thumbs up/down (not pairwise)
   - Lower data requirements

3. **"ORPO: Odds Ratio Preference Optimization"** (Hong et al., 2024)
   - Combines SFT + preference learning
   - Single-stage training
   - No reference model

### Hardware Optimization

**Model Size for Arc GPU (12GB VRAM):**
```python
# Maximum trainable model sizes on Arc GPU
MAX_MODEL_SIZES = {
    "fp16": "1.3B params",      # ~2.6GB model + ~8GB optimizer states
    "bf16": "1.3B params",      # Same as FP16
    "int8": "2B params",        # Quantized (inference only)
    "LoRA": "7B base model",    # Only train adapter (~10M params)
}

# Recommended: Use LoRA on 1.3B model
# Memory breakdown:
# - Base model (1.3B): ~2.6GB (BF16)
# - LoRA adapters: ~20MB (r=16)
# - Gradients: ~2.6GB
# - Optimizer states (AdamW): ~5.2GB
# - Activations (batch=4): ~1GB
# Total: ~11.4GB (fits in 12GB)
```

**NPU Optimization:**
- Use NPU for inference during validation
- INT8 quantization for production deployment
- Frees Arc GPU for pure training

### Implementation Steps

#### Week 1: Setup & Dataset Preparation

**File:** `02-ai-engine/rl_training/dpo_trainer.py`

```python
#!/usr/bin/env python3
"""
DPO Trainer - Direct Preference Optimization

Hardware-optimized for Intel Arc GPU (12GB VRAM)
Uses LoRA for parameter-efficient fine-tuning
Deploys to NPU for inference

Research Papers:
- Rafailov et al., "Direct Preference Optimization" (2023)
- Hu et al., "LoRA: Low-Rank Adaptation" (2021)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer
import intel_extension_for_pytorch as ipex  # Intel optimization

class HardwareOptimizedDPOTrainer:
    """DPO trainer optimized for Intel Arc GPU"""

    def __init__(
        self,
        model_name: str = "microsoft/phi-2",  # 2.7B params, fits with LoRA
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        use_arc_gpu: bool = True,
        use_npu_validation: bool = True
    ):
        self.device = "xpu" if use_arc_gpu else "cpu"  # Intel XPU = Arc GPU

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # BF16 for Arc GPU
            device_map="auto"
        )

        # Apply LoRA for memory efficiency
        if use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            )
            self.model = get_peft_model(self.model, peft_config)
            print(f"✓ LoRA enabled: {self.model.num_parameters()} trainable params")

        # Intel Arc GPU optimization
        if use_arc_gpu:
            self.model = ipex.optimize(
                self.model,
                dtype=torch.bfloat16,
                inplace=True
            )
            print("✓ Intel Arc GPU optimization applied")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.use_npu_validation = use_npu_validation

    def train(
        self,
        train_dataset,
        eval_dataset,
        output_dir: str = "./dpo_checkpoints",
        num_epochs: int = 3,
        batch_size: int = 2,  # Small batch for 12GB VRAM
        gradient_accumulation_steps: int = 8  # Effective batch = 16
    ):
        """
        Train with DPO loss

        Memory optimization:
        - Batch size = 2 (low VRAM)
        - Gradient accumulation = 8 (effective batch = 16)
        - Mixed precision (BF16)
        - LoRA (only 10M params trained)
        """

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=5e-5,
            bf16=True,  # BF16 for Arc GPU
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            warmup_steps=100,
            use_cpu=False,  # Force XPU/Arc GPU
            dataloader_num_workers=4,  # Parallel data loading
            remove_unused_columns=False,
            # Intel-specific optimizations
            gradient_checkpointing=True,  # Reduce memory
            optim="adamw_torch",
        )

        # DPO-specific config
        dpo_trainer = DPOTrainer(
            model=self.model,
            ref_model=None,  # Will create reference model automatically
            args=training_args,
            beta=0.1,  # DPO temperature
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            max_length=512,
            max_prompt_length=256,
        )

        # Train
        print("🚀 Starting DPO training on Intel Arc GPU...")
        dpo_trainer.train()

        # Save LoRA adapters
        self.model.save_pretrained(f"{output_dir}/final")
        print(f"✓ Model saved to {output_dir}/final")

        # Quantize for NPU deployment
        if self.use_npu_validation:
            self._deploy_to_npu(output_dir)

    def _deploy_to_npu(self, model_dir: str):
        """
        Quantize model to INT8 for NPU deployment

        Intel NPU supports:
        - INT8 quantization (49.4 TOPS)
        - INT4 quantization (custom kernels)
        """
        from neural_compressor import quantization
        from neural_compressor.config import PostTrainingQuantConfig

        # Load merged model (base + LoRA)
        merged_model = self.model.merge_and_unload()

        # Intel Neural Compressor quantization
        q_config = PostTrainingQuantConfig(
            backend="ipex",  # Intel backend
            approach="static",
            calibration_sampling_size=100,
        )

        quantized_model = quantization.fit(
            merged_model,
            q_config,
            calib_dataloader=self._get_calibration_data()
        )

        # Save INT8 model for NPU
        quantized_model.save(f"{model_dir}/npu_int8")
        print(f"✓ INT8 model saved for NPU: {model_dir}/npu_int8")
        print(f"  Expected throughput: ~40 TOPS on NPU")

# Dataset preparation
def prepare_dpo_dataset():
    """Load dataset from existing DPO generator"""
    from feedback.dpo_dataset_generator import DPODatasetGenerator

    generator = DPODatasetGenerator()
    dataset = generator.generate_dataset(
        min_pairs=1000,  # Start with 1K pairs
        include_ratings=True
    )

    # Convert to HuggingFace dataset format
    from datasets import Dataset
    hf_dataset = Dataset.from_list(dataset)

    # Train/eval split
    split = hf_dataset.train_test_split(test_size=0.1)
    return split['train'], split['test']

# Usage
if __name__ == "__main__":
    # Prepare data
    train_ds, eval_ds = prepare_dpo_dataset()

    # Train
    trainer = HardwareOptimizedDPOTrainer(
        model_name="microsoft/phi-2",  # 2.7B params, good quality
        use_lora=True,
        use_arc_gpu=True,
        use_npu_validation=True
    )

    trainer.train(
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        num_epochs=3,
        batch_size=2,
        gradient_accumulation_steps=8
    )
```

#### Week 2-3: Feedback Collection Infrastructure

**File:** `02-ai-engine/feedback/hitl_feedback_enhanced.py`

```python
#!/usr/bin/env python3
"""
Enhanced HITL Feedback Collection

Collects human feedback for DPO training:
- Thumbs up/down
- A/B comparisons
- Corrections
- Ratings (1-5 stars)
"""

import sqlite3
from datetime import datetime
from typing import Optional, Dict, List
import json

class EnhancedHITLFeedback:
    """Production-grade feedback collection"""

    def __init__(self, db_path: str = "~/.rag_index/hitl_feedback.db"):
        self.db_path = os.path.expanduser(db_path)
        self._init_db()

    def _init_db(self):
        """Initialize database with comprehensive schema"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Main feedback table
        c.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp REAL,
                query TEXT,
                response_a TEXT,
                response_b TEXT NULL,
                feedback_type TEXT,  -- 'thumbs', 'comparison', 'correction', 'rating'
                feedback_value TEXT,  -- JSON: {"thumbs": "up"} or {"chosen": "a"}
                context TEXT NULL,
                metadata TEXT NULL
            )
        ''')

        # Agent performance tracking
        c.execute('''
            CREATE TABLE IF NOT EXISTS agent_performance (
                agent_id TEXT,
                task_type TEXT,
                success_rate REAL,
                avg_rating REAL,
                total_interactions INTEGER,
                last_updated REAL,
                PRIMARY KEY (agent_id, task_type)
            )
        ''')

        conn.commit()
        conn.close()

    def record_thumbs(
        self,
        query: str,
        response: str,
        thumbs_up: bool,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ):
        """Record thumbs up/down feedback"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            INSERT INTO feedback (
                session_id, timestamp, query, response_a,
                feedback_type, feedback_value
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            session_id or "default",
            datetime.now().timestamp(),
            query,
            response,
            "thumbs",
            json.dumps({"thumbs": "up" if thumbs_up else "down", "agent_id": agent_id})
        ))

        conn.commit()
        conn.close()

        # Update agent performance
        if agent_id:
            self._update_agent_performance(agent_id, "general", thumbs_up)

    def record_comparison(
        self,
        query: str,
        response_a: str,
        response_b: str,
        chosen: str,  # "a" or "b"
        session_id: Optional[str] = None
    ):
        """Record A/B comparison preference"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            INSERT INTO feedback (
                session_id, timestamp, query, response_a, response_b,
                feedback_type, feedback_value
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id or "default",
            datetime.now().timestamp(),
            query,
            response_a,
            response_b,
            "comparison",
            json.dumps({"chosen": chosen})
        ))

        conn.commit()
        conn.close()

    def get_dpo_pairs(self, min_pairs: int = 100) -> List[Dict]:
        """
        Get preference pairs for DPO training

        Returns:
            List of {"prompt": ..., "chosen": ..., "rejected": ...}
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        pairs = []

        # 1. From A/B comparisons
        c.execute('''
            SELECT query, response_a, response_b, feedback_value
            FROM feedback
            WHERE feedback_type = 'comparison'
            LIMIT ?
        ''', (min_pairs,))

        for row in c.fetchall():
            query, resp_a, resp_b, value_json = row
            value = json.loads(value_json)
            chosen = resp_a if value['chosen'] == 'a' else resp_b
            rejected = resp_b if value['chosen'] == 'a' else resp_a

            pairs.append({
                "prompt": query,
                "chosen": chosen,
                "rejected": rejected
            })

        # 2. From thumbs (up vs down)
        # Group by query, find up vs down responses
        c.execute('''
            SELECT f1.query, f1.response_a, f2.response_a
            FROM feedback f1
            JOIN feedback f2 ON f1.query = f2.query
            WHERE f1.feedback_type = 'thumbs'
              AND f2.feedback_type = 'thumbs'
              AND json_extract(f1.feedback_value, '$.thumbs') = 'up'
              AND json_extract(f2.feedback_value, '$.thumbs') = 'down'
            LIMIT ?
        ''', (min_pairs - len(pairs),))

        for row in c.fetchall():
            query, chosen, rejected = row
            pairs.append({
                "prompt": query,
                "chosen": chosen,
                "rejected": rejected
            })

        conn.close()
        return pairs

    def _update_agent_performance(self, agent_id: str, task_type: str, success: bool):
        """Update agent performance metrics"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Get current performance
        c.execute('''
            SELECT success_rate, total_interactions
            FROM agent_performance
            WHERE agent_id = ? AND task_type = ?
        ''', (agent_id, task_type))

        row = c.fetchone()
        if row:
            old_rate, total = row
            new_total = total + 1
            new_rate = (old_rate * total + (1 if success else 0)) / new_total

            c.execute('''
                UPDATE agent_performance
                SET success_rate = ?, total_interactions = ?, last_updated = ?
                WHERE agent_id = ? AND task_type = ?
            ''', (new_rate, new_total, datetime.now().timestamp(), agent_id, task_type))
        else:
            c.execute('''
                INSERT INTO agent_performance
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (agent_id, task_type, 1.0 if success else 0.0, 0.0, 1, datetime.now().timestamp()))

        conn.commit()
        conn.close()
```

#### Week 4-6: Training & Validation

**Training Script:** `scripts/train_dpo.py`

```python
#!/usr/bin/env python3
"""
DPO Training Runner

Optimized for Intel Arc GPU (12GB VRAM)
Deploys to Intel NPU for inference
"""

import argparse
from rl_training.dpo_trainer import HardwareOptimizedDPOTrainer
from feedback.hitl_feedback_enhanced import EnhancedHITLFeedback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/phi-2", help="Base model")
    parser.add_argument("--min-pairs", type=int, default=1000, help="Min training pairs")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--output", default="./dpo_models", help="Output directory")
    args = parser.parse_args()

    # Collect feedback data
    print("=" * 80)
    print("PHASE 1: Collecting Feedback Data")
    print("=" * 80)

    feedback = EnhancedHITLFeedback()
    pairs = feedback.get_dpo_pairs(min_pairs=args.min_pairs)

    if len(pairs) < args.min_pairs:
        print(f"⚠️  Warning: Only {len(pairs)} pairs available (requested {args.min_pairs})")
        print(f"   Consider collecting more feedback before training")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    print(f"✓ Collected {len(pairs)} preference pairs")

    # Convert to HuggingFace dataset
    from datasets import Dataset
    dataset = Dataset.from_list(pairs)
    split = dataset.train_test_split(test_size=0.1)

    # Train
    print("\n" + "=" * 80)
    print("PHASE 2: DPO Training on Intel Arc GPU")
    print("=" * 80)

    trainer = HardwareOptimizedDPOTrainer(
        model_name=args.model,
        use_lora=True,
        lora_r=args.lora_r,
        use_arc_gpu=True,
        use_npu_validation=True
    )

    trainer.train(
        train_dataset=split['train'],
        eval_dataset=split['test'],
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum
    )

    print("\n" + "=" * 80)
    print("✅ DPO Training Complete!")
    print("=" * 80)
    print(f"Models saved to: {args.output}/")
    print(f"  - LoRA adapters: {args.output}/final/")
    print(f"  - INT8 for NPU: {args.output}/npu_int8/")
    print(f"\nNPU Deployment:")
    print(f"  Load INT8 model for inference on NPU (49.4 TOPS)")
    print(f"  Expected latency: <50ms per query")

if __name__ == "__main__":
    main()
```

### Expected Outcomes

**Week 6 Deliverables:**
- ✅ DPO training pipeline fully functional
- ✅ Trained on 1K+ preference pairs
- ✅ Model runs on Arc GPU (training) + NPU (inference)
- ✅ +15-25% agent quality improvement

**Metrics:**
- Training time: ~4-8 hours for 1.3B model (3 epochs)
- Memory usage: ~11GB VRAM on Arc GPU
- Inference latency: ~50ms on NPU (INT8)
- Throughput: ~40 TOPS on NPU

---

## PHASE 2: Self-RAG with Reflection (Weeks 7-12)

### Goal: Add Iterative Refinement to RAG Pipeline

**Why Self-RAG:**
- Handles complex multi-step queries
- Self-assessment of retrieval quality
- Adaptive retrieval (retrieve only when needed)
- Low compute overhead (reflection = lightweight LLM call)

### Research Papers

1. **"Self-RAG: Learning to Retrieve, Generate, and Critique"** (Asai et al., 2023)
   - Main paper
   - Reflection tokens: [Retrieval], [Relevance], [Support], [Utility]
   - Self-assessment framework
   - Critic model for filtering

2. **"FLARE: Active Retrieval Augmented Generation"** (Jiang et al., 2023)
   - Lookahead-based retrieval
   - Only retrieve when uncertain
   - Cost-efficient

3. **"Adaptive-RAG: Learning to Adapt"** (Jeong et al., 2024)
   - Query complexity classifier
   - 3 strategies: no retrieval, single, iterative

### Hardware Optimization

**Model Deployment:**
```python
# Self-RAG components and hardware allocation

SELF_RAG_HARDWARE = {
    "retrieval_embedder": {
        "model": "all-MiniLM-L6-v2",  # 384-dim embeddings
        "hardware": "NPU",             # Continuous embeddings on NPU
        "quantization": "INT8",
        "throughput": "~1000 embeds/sec",
        "latency": "~1ms per embed"
    },

    "critic_model": {
        "model": "microsoft/phi-1.5",  # 1.3B params for critique
        "hardware": "Arc GPU",          # Critic runs on GPU
        "quantization": "BF16",
        "throughput": "~10 critiques/sec",
        "latency": "~100ms per critique"
    },

    "generator_model": {
        "model": "microsoft/phi-2",    # 2.7B params for generation
        "hardware": "Arc GPU",          # Main generation on GPU
        "quantization": "BF16 (train), INT8 (deploy)",
        "throughput": "~5 responses/sec",
        "latency": "~200ms per response"
    },

    "reranker": {
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "hardware": "NCS2 stick #1",   # Reranking on NCS2
        "quantization": "INT8",
        "throughput": "~50 pairs/sec",
        "latency": "~20ms per pair"
    },

    "chroma_db": {
        "backend": "ChromaDB",
        "hardware": "CPU + AVX-512",   # Vector search on CPU
        "index": "HNSW",
        "throughput": "~500 searches/sec",
        "latency": "~2ms per search"
    }
}

# Total RAG pipeline latency breakdown:
# 1. Embed query (NPU): 1ms
# 2. Vector search (CPU/AVX-512): 2ms
# 3. Rerank top-20 (NCS2): 20ms
# 4. Critic assessment (Arc GPU): 100ms
# 5. Generate response (Arc GPU): 200ms
# TOTAL: ~323ms end-to-end (< 500ms target ✓)
```

**NPU Optimization for Embeddings:**
```python
# Intel NPU optimization for sentence embeddings

import openvino as ov
from sentence_transformers import SentenceTransformer

def convert_embedder_to_npu():
    """
    Convert sentence-transformers to OpenVINO IR for NPU

    INT8 quantization achieves:
    - 4x memory reduction
    - 3-5x speedup on NPU
    - <1% accuracy loss
    """
    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Export to ONNX
    dummy_input = torch.randn(1, 128)  # Max seq length
    torch.onnx.export(
        model,
        dummy_input,
        "embedder.onnx",
        input_names=['input_ids'],
        output_names=['embeddings'],
        dynamic_axes={'input_ids': {0: 'batch', 1: 'seq'}}
    )

    # Convert to OpenVINO IR
    ov_model = ov.convert_model("embedder.onnx")

    # Quantize to INT8 for NPU
    from openvino.tools import mo
    quantized_model = mo.convert_model(
        ov_model,
        compress_to_fp16=False,
        compress_to_int8=True  # INT8 for NPU
    )

    # Save for NPU deployment
    ov.serialize(quantized_model, "embedder_npu_int8.xml")

    print("✓ Embedder converted to INT8 for NPU")
    print("  Expected throughput: ~1000 embeddings/sec")
    print("  Expected latency: ~1ms per embed")

# NPU inference
core = ov.Core()
npu_model = core.read_model("embedder_npu_int8.xml")
compiled = core.compile_model(npu_model, "NPU")

def embed_on_npu(text: str) -> np.ndarray:
    """Fast embedding on NPU"""
    tokens = tokenizer(text, return_tensors="np")
    result = compiled([tokens['input_ids']])[0]
    return result  # 384-dim embedding
```

### Implementation Steps

#### Week 7-8: Reflection Framework

**File:** `02-ai-engine/deep_thinking_rag/self_rag_engine.py`

```python
#!/usr/bin/env python3
"""
Self-RAG Engine with Reflection Tokens

Implements reflection-based retrieval:
1. Assess if retrieval is needed
2. Retrieve documents
3. Critique relevance
4. Generate with support assessment
5. Evaluate utility

Hardware: Critic on Arc GPU, Embedder on NPU
"""

import enum
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

class ReflectionToken(enum.Enum):
    """Reflection tokens for self-assessment"""
    RETRIEVAL_NEEDED = "[Retrieval]"
    RETRIEVAL_NOT_NEEDED = "[No Retrieval]"
    RELEVANT = "[Relevant]"
    IRRELEVANT = "[Irrelevant]"
    SUPPORTED = "[Supported]"
    NOT_SUPPORTED = "[Not Supported]"
    USEFUL = "[Useful]"
    NOT_USEFUL = "[Not Useful]"

@dataclass
class RetrievalDecision:
    """Decision about whether to retrieve"""
    should_retrieve: bool
    confidence: float
    reasoning: str

@dataclass
class CritiqueResult:
    """Critique of retrieved documents"""
    relevant_docs: List[int]  # Indices of relevant docs
    relevance_scores: List[float]
    overall_quality: float
    reasoning: str

class SelfRAGEngine:
    """
    Self-assessing RAG with reflection

    Pipeline:
    1. Query → Retrieval Decision (critic)
    2. If yes → Retrieve docs (NPU embedder + ChromaDB)
    3. Critique relevance (critic)
    4. Filter irrelevant docs
    5. Generate response (main model)
    6. Assess support & utility (critic)
    7. If utility low → iterate
    """

    def __init__(
        self,
        rag_system,  # EnhancedRAGSystem
        critic_model_name: str = "microsoft/phi-1.5",  # 1.3B critic
        generator_model_name: str = "microsoft/phi-2",  # 2.7B generator
        use_npu_embeddings: bool = True,
        max_iterations: int = 3
    ):
        self.rag = rag_system
        self.max_iterations = max_iterations

        # Load critic model (Arc GPU)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.critic = AutoModelForCausalLM.from_pretrained(
            critic_model_name,
            torch_dtype=torch.bfloat16,
            device_map="xpu"  # Arc GPU
        )
        self.critic_tokenizer = AutoTokenizer.from_pretrained(critic_model_name)

        # Load generator model (Arc GPU)
        self.generator = AutoModelForCausalLM.from_pretrained(
            generator_model_name,
            torch_dtype=torch.bfloat16,
            device_map="xpu"
        )
        self.gen_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)

        # NPU embeddings
        if use_npu_embeddings:
            self._init_npu_embedder()

    def _init_npu_embedder(self):
        """Initialize NPU-optimized embedder"""
        import openvino as ov
        core = ov.Core()
        self.npu_embedder = core.compile_model(
            core.read_model("embedder_npu_int8.xml"),
            "NPU"
        )
        print("✓ NPU embedder loaded (INT8, 49.4 TOPS)")

    def query(
        self,
        query: str,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Self-RAG query with reflection

        Returns:
            {
                "response": str,
                "iterations": int,
                "retrieved_docs": List[str],
                "reflection_trace": List[Dict]
            }
        """
        reflection_trace = []
        iteration = 0
        accumulated_context = []

        while iteration < self.max_iterations:
            iteration += 1

            # STEP 1: Should we retrieve?
            decision = self._assess_retrieval_need(query, accumulated_context)
            reflection_trace.append({
                "step": "retrieval_decision",
                "iteration": iteration,
                "should_retrieve": decision.should_retrieve,
                "reasoning": decision.reasoning
            })

            if not decision.should_retrieve:
                # Generate without retrieval
                response = self._generate(query, accumulated_context)
                break

            # STEP 2: Retrieve documents
            docs = self._retrieve(query)
            reflection_trace.append({
                "step": "retrieval",
                "iteration": iteration,
                "num_docs": len(docs)
            })

            # STEP 3: Critique relevance
            critique = self._critique_relevance(query, docs)
            reflection_trace.append({
                "step": "critique",
                "iteration": iteration,
                "relevant_count": len(critique.relevant_docs),
                "quality": critique.overall_quality
            })

            # STEP 4: Filter and accumulate
            relevant_docs = [docs[i] for i in critique.relevant_docs]
            accumulated_context.extend(relevant_docs)

            # STEP 5: Generate response
            response = self._generate(query, accumulated_context)

            # STEP 6: Assess utility
            utility = self._assess_utility(query, response, accumulated_context)
            reflection_trace.append({
                "step": "utility_assessment",
                "iteration": iteration,
                "utility": utility.score,
                "reasoning": utility.reasoning
            })

            # STEP 7: Decide if we're done
            if utility.score > 0.7:  # Good enough
                break
            elif iteration >= self.max_iterations:
                break
            # Otherwise, continue to next iteration

        return {
            "response": response,
            "iterations": iteration,
            "retrieved_docs": accumulated_context,
            "reflection_trace": reflection_trace
        }

    def _assess_retrieval_need(
        self,
        query: str,
        context: List[str]
    ) -> RetrievalDecision:
        """
        Assess if retrieval is needed

        Critic prompt:
        "Given the query '{query}' and current context, do you need to retrieve
        more information? Answer with [Retrieval] or [No Retrieval] and explain."
        """
        prompt = f"""<|system|>You are a retrieval assessment critic. Decide if retrieval is needed.<|end|>
<|user|>
Query: {query}

Current context: {len(context)} documents already retrieved.

Should we retrieve more documents? Respond with [Retrieval] or [No Retrieval], then explain your reasoning.<|end|>
<|assistant|>"""

        inputs = self.critic_tokenizer(prompt, return_tensors="pt").to("xpu")
        outputs = self.critic.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,  # Low temp for consistency
            do_sample=False
        )

        response = self.critic_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parse reflection token
        should_retrieve = ReflectionToken.RETRIEVAL_NEEDED.value in response

        # Extract reasoning
        reasoning = response.split(ReflectionToken.RETRIEVAL_NEEDED.value if should_retrieve else ReflectionToken.RETRIEVAL_NOT_NEEDED.value)[-1].strip()

        return RetrievalDecision(
            should_retrieve=should_retrieve,
            confidence=0.9 if "[Retrieval]" in response or "[No Retrieval]" in response else 0.5,
            reasoning=reasoning
        )

    def _retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """Retrieve documents using NPU embeddings"""
        # Embed on NPU
        if hasattr(self, 'npu_embedder'):
            # NPU embedding (~1ms)
            query_embedding = self._embed_on_npu(query)
        else:
            # Fallback to CPU
            query_embedding = self.rag.embed(query)

        # Vector search (CPU/AVX-512, ~2ms)
        results = self.rag.search(
            query_embedding=query_embedding,
            top_k=top_k * 2  # Retrieve 2x, will filter
        )

        # Rerank on NCS2 (~20ms)
        reranked = self.rag.rerank(query, results, top_k=top_k)

        return [r.text for r in reranked]

    def _critique_relevance(
        self,
        query: str,
        docs: List[str]
    ) -> CritiqueResult:
        """
        Critique document relevance

        Critic assesses each doc with [Relevant] or [Irrelevant]
        """
        relevant_indices = []
        relevance_scores = []

        for i, doc in enumerate(docs):
            prompt = f"""<|system|>You are a document relevance critic.<|end|>
<|user|>
Query: {query}

Document: {doc[:500]}...

Is this document relevant to the query? Respond with [Relevant] or [Irrelevant], then explain.<|end|>
<|assistant|>"""

            inputs = self.critic_tokenizer(prompt, return_tensors="pt").to("xpu")
            outputs = self.critic.generate(**inputs, max_new_tokens=50, temperature=0.1)
            response = self.critic_tokenizer.decode(outputs[0], skip_special_tokens=True)

            if ReflectionToken.RELEVANT.value in response:
                relevant_indices.append(i)
                relevance_scores.append(0.8)  # Could extract confidence from response

        return CritiqueResult(
            relevant_docs=relevant_indices,
            relevance_scores=relevance_scores,
            overall_quality=len(relevant_indices) / len(docs) if docs else 0.0,
            reasoning=f"Found {len(relevant_indices)}/{len(docs)} relevant documents"
        )

    def _generate(self, query: str, context: List[str]) -> str:
        """Generate response from query + context"""
        context_str = "\n\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(context)])

        prompt = f"""<|system|>You are a helpful assistant. Use the provided context to answer the query.<|end|>
<|user|>
Context:
{context_str}

Query: {query}<|end|>
<|assistant|>"""

        inputs = self.gen_tokenizer(prompt, return_tensors="pt").to("xpu")
        outputs = self.generator.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )

        response = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("<|assistant|>")[-1].strip()

    def _assess_utility(
        self,
        query: str,
        response: str,
        context: List[str]
    ) -> Dict:
        """
        Assess if response is useful

        Returns utility score and reasoning
        """
        prompt = f"""<|system|>You are a response quality critic.<|end|>
<|user|>
Query: {query}

Response: {response}

Is this response useful and well-supported by the context? Respond with [Useful] or [Not Useful], then explain.<|end|>
<|assistant|>"""

        inputs = self.critic_tokenizer(prompt, return_tensors="pt").to("xpu")
        outputs = self.critic.generate(**inputs, max_new_tokens=100, temperature=0.1)
        critique = self.critic_tokenizer.decode(outputs[0], skip_special_tokens=True)

        score = 0.9 if ReflectionToken.USEFUL.value in critique else 0.3
        reasoning = critique.split("[Useful]" if score > 0.5 else "[Not Useful]")[-1].strip()

        return {
            "score": score,
            "reasoning": reasoning
        }
```

#### Week 9-10: Adaptive Retrieval Strategy

**File:** `02-ai-engine/deep_thinking_rag/adaptive_strategy_selector.py`

```python
#!/usr/bin/env python3
"""
Adaptive Retrieval Strategy Selector

Based on query difficulty, selects:
- No retrieval (simple facts)
- Single retrieval (straightforward questions)
- Iterative retrieval (complex multi-hop questions)

Research: "Adaptive-RAG" (Jeong et al., 2024)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from dataclasses import dataclass
from enum import Enum

class RetrievalStrategy(Enum):
    """Retrieval strategies"""
    NO_RETRIEVAL = "no_retrieval"
    SINGLE_RETRIEVAL = "single_retrieval"
    ITERATIVE_RETRIEVAL = "iterative_retrieval"

@dataclass
class StrategyDecision:
    """Strategy selection result"""
    strategy: RetrievalStrategy
    confidence: float
    reasoning: str

class DifficultyClassifier(nn.Module):
    """
    Query difficulty classifier

    Classifies queries into:
    - Easy (no retrieval needed)
    - Medium (single retrieval)
    - Hard (iterative retrieval)
    """

    def __init__(self, embedding_dim: int = 384):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)  # 3 classes
        self.dropout = nn.Dropout(0.1)

    def forward(self, embeddings):
        x = torch.relu(self.fc1(embeddings))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)
        return logits

class AdaptiveStrategySelector:
    """
    Selects retrieval strategy based on query difficulty

    Small classifier (500K params) runs on NPU
    """

    def __init__(
        self,
        embedder_model: str = "all-MiniLM-L6-v2",
        use_npu: bool = True
    ):
        # Load embedder
        self.tokenizer = AutoTokenizer.from_pretrained(embedder_model)
        self.embedder = AutoModel.from_pretrained(embedder_model)

        # Load classifier
        self.classifier = DifficultyClassifier(embedding_dim=384)

        # Try to load trained weights
        try:
            self.classifier.load_state_dict(torch.load("difficulty_classifier.pth"))
            print("✓ Loaded trained difficulty classifier")
        except:
            print("⚠️  No trained classifier found, using untrained (train first!)")

        # Deploy to NPU
        if use_npu:
            self._deploy_to_npu()

    def _deploy_to_npu(self):
        """Deploy classifier to NPU for low-latency inference"""
        # Convert to OpenVINO IR
        import openvino as ov

        # Export to ONNX
        dummy_input = torch.randn(1, 384)
        torch.onnx.export(
            self.classifier,
            dummy_input,
            "classifier.onnx",
            input_names=['embeddings'],
            output_names=['logits']
        )

        # Convert to OpenVINO
        ov_model = ov.convert_model("classifier.onnx")

        # Quantize to INT8
        from openvino.tools import mo
        quantized = mo.convert_model(ov_model, compress_to_int8=True)

        # Compile for NPU
        core = ov.Core()
        self.npu_classifier = core.compile_model(quantized, "NPU")
        print("✓ Difficulty classifier deployed to NPU (INT8)")

    def select_strategy(self, query: str) -> StrategyDecision:
        """
        Select retrieval strategy for query

        Latency: ~2ms total (embedding on NPU + classification on NPU)
        """
        # Embed query (NPU, ~1ms)
        tokens = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            embedding = self.embedder(**tokens).last_hidden_state.mean(dim=1).squeeze()

        # Classify difficulty (NPU, ~1ms)
        if hasattr(self, 'npu_classifier'):
            logits = self.npu_classifier([embedding.numpy()])[0]
            probs = torch.softmax(torch.tensor(logits), dim=-1)
        else:
            with torch.no_grad():
                logits = self.classifier(embedding.unsqueeze(0))
                probs = torch.softmax(logits, dim=-1).squeeze()

        # Map to strategy
        class_idx = torch.argmax(probs).item()
        confidence = probs[class_idx].item()

        strategy_map = {
            0: RetrievalStrategy.NO_RETRIEVAL,
            1: RetrievalStrategy.SINGLE_RETRIEVAL,
            2: RetrievalStrategy.ITERATIVE_RETRIEVAL
        }

        strategy = strategy_map[class_idx]

        reasoning_map = {
            RetrievalStrategy.NO_RETRIEVAL: "Simple factual query, answer from model knowledge",
            RetrievalStrategy.SINGLE_RETRIEVAL: "Straightforward question, single retrieval sufficient",
            RetrievalStrategy.ITERATIVE_RETRIEVAL: "Complex multi-hop query, needs iterative refinement"
        }

        return StrategyDecision(
            strategy=strategy,
            confidence=confidence,
            reasoning=reasoning_map[strategy]
        )

# Training script for difficulty classifier
def train_difficulty_classifier():
    """
    Train difficulty classifier on labeled queries

    Dataset format:
    [
        {"query": "What is the capital of France?", "difficulty": 0},  # Easy
        {"query": "How does photosynthesis work?", "difficulty": 1},  # Medium
        {"query": "Explain the relationship between quantum mechanics and general relativity", "difficulty": 2}  # Hard
    ]
    """
    from datasets import load_dataset
    import torch.optim as optim

    # Load dataset (create manually or use synthetic generation)
    # For now, we'll use a simple heuristic-based labeling:
    # - Short queries with simple keywords → Easy
    # - Medium length with specific questions → Medium
    # - Long queries with multiple concepts → Hard

    dataset = generate_synthetic_difficulty_dataset(n_samples=5000)

    # Split
    train_data, val_data = dataset[:-500], dataset[-500:]

    # Initialize
    selector = AdaptiveStrategySelector(use_npu=False)
    optimizer = optim.Adam(selector.classifier.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Train
    epochs = 10
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_data:
            # Embed
            tokens = selector.tokenizer(batch['query'], return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                embedding = selector.embedder(**tokens).last_hidden_state.mean(dim=1)

            # Forward
            logits = selector.classifier(embedding)
            loss = criterion(logits, torch.tensor([batch['difficulty']]))

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy
            pred = torch.argmax(logits, dim=-1).item()
            correct += (pred == batch['difficulty'])
            total += 1

        # Validation
        val_acc = evaluate_classifier(selector, val_data)

        print(f"Epoch {epoch+1}/{epochs}: Loss={total_loss/total:.4f}, Train Acc={correct/total:.4f}, Val Acc={val_acc:.4f}")

    # Save
    torch.save(selector.classifier.state_dict(), "difficulty_classifier.pth")
    print("✓ Classifier saved to difficulty_classifier.pth")

def generate_synthetic_difficulty_dataset(n_samples: int = 5000):
    """
    Generate synthetic dataset for difficulty classification

    Uses heuristics:
    - Easy: Short, simple keywords
    - Medium: Specific questions, moderate length
    - Hard: Long, multiple concepts, complex reasoning
    """
    import random

    easy_templates = [
        "What is {noun}?",
        "Define {noun}",
        "Who is {person}?",
        "Where is {place}?",
    ]

    medium_templates = [
        "How does {process} work?",
        "Explain {concept}",
        "What are the benefits of {noun}?",
        "Compare {noun1} and {noun2}",
    ]

    hard_templates = [
        "Analyze the relationship between {concept1} and {concept2} in the context of {domain}",
        "What are the implications of {event} on {outcome}, considering {factor1} and {factor2}?",
        "Synthesize information about {topic} from multiple perspectives including {view1}, {view2}, and {view3}",
    ]

    dataset = []

    for _ in range(n_samples):
        difficulty = random.choice([0, 1, 2])

        if difficulty == 0:
            template = random.choice(easy_templates)
            query = template.format(
                noun=random.choice(["Python", "DNA", "gravity", "democracy"]),
                person=random.choice(["Einstein", "Tesla", "Curie"]),
                place=random.choice(["Paris", "Mount Everest", "Amazon"])
            )
        elif difficulty == 1:
            template = random.choice(medium_templates)
            query = template.format(
                process=random.choice(["photosynthesis", "machine learning", "encryption"]),
                concept=random.choice(["quantum computing", "blockchain", "neural networks"]),
                noun=random.choice(["solar panels", "vaccines", "electric cars"]),
                noun1=random.choice(["TCP", "UDP"]),
                noun2=random.choice(["HTTP", "HTTPS"])
            )
        else:
            template = random.choice(hard_templates)
            query = template.format(
                concept1=random.choice(["quantum mechanics", "general relativity"]),
                concept2=random.choice(["thermodynamics", "information theory"]),
                domain=random.choice(["physics", "computer science", "biology"]),
                event=random.choice(["climate change", "AI advancement", "genetic engineering"]),
                outcome=random.choice(["society", "economy", "environment"]),
                factor1=random.choice(["ethics", "policy", "technology"]),
                factor2=random.choice(["economics", "culture", "science"]),
                topic=random.choice(["sustainable energy", "space exploration", "bioethics"]),
                view1=random.choice(["scientific", "economic", "ethical"]),
                view2=random.choice(["political", "social", "technological"]),
                view3=random.choice(["environmental", "cultural", "historical"])
            )

        dataset.append({"query": query, "difficulty": difficulty})

    return dataset
```

#### Week 11-12: Integration & Testing

**Complete Self-RAG Pipeline:**

```python
#!/usr/bin/env python3
"""
Complete Self-RAG Pipeline with Adaptive Strategy

Hardware distribution:
- NPU: Embeddings + difficulty classifier (~3ms total)
- Arc GPU: Critic + generator (~300ms total)
- NCS2: Reranking (~20ms)
- CPU/AVX-512: Vector search (~2ms)

Total latency: ~325ms for simple queries, ~500-800ms for iterative
"""

from deep_thinking_rag.self_rag_engine import SelfRAGEngine
from deep_thinking_rag.adaptive_strategy_selector import AdaptiveStrategySelector, RetrievalStrategy

class CompleteSelfRAG:
    """
    Production Self-RAG with hardware optimization
    """

    def __init__(self, rag_system):
        # Strategy selector (NPU)
        self.strategy_selector = AdaptiveStrategySelector(use_npu=True)

        # Self-RAG engine (Arc GPU + NPU + NCS2)
        self.self_rag = SelfRAGEngine(
            rag_system=rag_system,
            use_npu_embeddings=True
        )

    def query(self, query: str) -> Dict:
        """
        Adaptive Self-RAG query

        Steps:
        1. Classify query difficulty (NPU, ~2ms)
        2. Select retrieval strategy
        3. Execute with Self-RAG engine
        """
        import time
        start = time.time()

        # STEP 1: Strategy selection (NPU)
        strategy_decision = self.strategy_selector.select_strategy(query)
        strategy_time = time.time() - start

        print(f"Strategy: {strategy_decision.strategy.value} (confidence: {strategy_decision.confidence:.2f})")
        print(f"  Reasoning: {strategy_decision.reasoning}")
        print(f"  Latency: {strategy_time*1000:.1f}ms")

        # STEP 2: Execute based on strategy
        if strategy_decision.strategy == RetrievalStrategy.NO_RETRIEVAL:
            # Generate directly without retrieval
            response = self.self_rag._generate(query, [])
            result = {
                "response": response,
                "strategy": "no_retrieval",
                "iterations": 0,
                "retrieved_docs": [],
                "latency_ms": (time.time() - start) * 1000
            }

        elif strategy_decision.strategy == RetrievalStrategy.SINGLE_RETRIEVAL:
            # Single retrieval pass
            docs = self.self_rag._retrieve(query)
            critique = self.self_rag._critique_relevance(query, docs)
            relevant_docs = [docs[i] for i in critique.relevant_docs]
            response = self.self_rag._generate(query, relevant_docs)

            result = {
                "response": response,
                "strategy": "single_retrieval",
                "iterations": 1,
                "retrieved_docs": relevant_docs,
                "latency_ms": (time.time() - start) * 1000
            }

        else:  # ITERATIVE_RETRIEVAL
            # Full iterative Self-RAG
            result = self.self_rag.query(query)
            result["strategy"] = "iterative_retrieval"
            result["latency_ms"] = (time.time() - start) * 1000

        print(f"Total latency: {result['latency_ms']:.1f}ms")

        return result

# Usage example
if __name__ == "__main__":
    from enhanced_rag_system import EnhancedRAGSystem

    # Initialize base RAG
    rag = EnhancedRAGSystem(enable_reranking=True)

    # Index some documents
    rag.index_directory("./knowledge_base")

    # Create Self-RAG
    self_rag = CompleteSelfRAG(rag)

    # Test queries
    test_queries = [
        "What is the capital of France?",  # Should use NO_RETRIEVAL
        "How do I optimize SQL queries?",  # Should use SINGLE_RETRIEVAL
        "Explain the relationship between quantum entanglement and information theory, considering both Copenhagen and many-worlds interpretations",  # Should use ITERATIVE_RETRIEVAL
    ]

    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"Query: {query}")
        print("=" * 80)

        result = self_rag.query(query)

        print(f"\nResponse: {result['response'][:200]}...")
        print(f"Strategy: {result['strategy']}")
        print(f"Iterations: {result['iterations']}")
        print(f"Docs retrieved: {len(result['retrieved_docs'])}")
        print(f"Latency: {result['latency_ms']:.1f}ms")
```

### Expected Outcomes

**Week 12 Deliverables:**
- ✅ Self-RAG with reflection fully implemented
- ✅ Adaptive strategy selector (NPU-optimized)
- ✅ Hardware-distributed pipeline (NPU+Arc+NCS2+CPU)
- ✅ +10-20% RAG accuracy on complex queries
- ✅ <500ms latency for most queries

**Performance Metrics:**
- Simple queries: ~200ms (no retrieval)
- Medium queries: ~325ms (single retrieval)
- Complex queries: ~500-800ms (iterative)
- NPU utilization: ~30-40% (embeddings + classifier)
- Arc GPU utilization: ~60-70% (critic + generator)

---

## [CONTINUED IN NEXT MESSAGE - This is getting long!]

**Document Status:** Part 1 of 4-Phase Implementation Plan
**Completed:** Phase 1 (DPO) + Phase 2 (Self-RAG) detailed
**Remaining:** Phase 3 (PPO + MoE) + Phase 4 (Meta-Learning + Evaluation)

Shall I continue with Phases 3-4 covering:
- PPO training with cloud GPU requirements
- Learned MoE routing
- Meta-learning (MAML)
- Comprehensive evaluation framework

?

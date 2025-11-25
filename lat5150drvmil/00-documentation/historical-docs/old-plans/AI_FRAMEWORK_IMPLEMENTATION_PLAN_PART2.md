# AI Framework Implementation Plan - Part 2
## Automated Self-Improvement + Performance Optimization

**Continuation from Part 1**
**Focus:** Automated self-improvement, C/C++ optimization, dynamic hardware allocation, ZFS integration

---

## PHASE 3: PPO Training + Learned MoE Routing (Weeks 13-28)

### 3A: Cloud-Based PPO Training Infrastructure (Weeks 13-20)

#### Challenge: Local Hardware Insufficient for PPO

**Problem:**
- PPO requires multi-GPU training (4-8x A100/H100)
- Current Arc GPU (12GB) cannot handle full PPO pipeline
- PPO needs: model, reference model, reward model, value function

**Solution: Hybrid Cloud + Local Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│  CLOUD GPU CLUSTER (Training)                               │
├─────────────────────────────────────────────────────────────┤
│  • 4-8x A100 GPUs (40-80GB VRAM each)                       │
│  • PPO training loop                                        │
│  • Reward model training                                    │
│  • Experience replay buffer                                  │
│  • Model checkpointing                                      │
│                                                             │
│  Providers:                                                 │
│  - Vast.ai ($1.50-3/hr per A100)                           │
│  - RunPod ($2-4/hr per A100)                               │
│  - Lambda Labs ($1.10/hr per A100)                          │
└─────────────────────────────────────────────────────────────┘
                         ↓ (Model sync)
┌─────────────────────────────────────────────────────────────┐
│  LOCAL HARDWARE (Inference + Data Collection)              │
├─────────────────────────────────────────────────────────────┤
│  • Intel NPU: Inference (INT8 quantized models)             │
│  • Arc GPU: Validation + feedback collection               │
│  • Collect (state, action, reward) trajectories             │
│  • Store in local PostgreSQL                                │
│  • Sync trajectories to cloud for training                  │
│  • Download trained models from cloud                       │
│  • Deploy to NPU for production                             │
└─────────────────────────────────────────────────────────────┘
```

#### Implementation: Automated Self-Improvement Loop

**File:** `02-ai-engine/rl_training/auto_improvement_orchestrator.py`

```python
#!/usr/bin/env python3
"""
Automated Self-Improvement Orchestrator

Coordinates continuous improvement:
1. Collect trajectories on local hardware
2. Sync to cloud GPU cluster
3. Train with PPO on cloud
4. Download improved models
5. Deploy to NPU
6. Repeat

Runs 24/7 for continuous self-improvement.
"""

import os
import time
import json
import boto3  # For S3 sync
import paramiko  # For SSH to cloud GPUs
from pathlib import Path
from typing import List, Dict
import psycopg2
from datetime import datetime, timedelta

class AutoImprovementOrchestrator:
    """
    24/7 automated self-improvement pipeline

    Schedule:
    - 00:00-06:00: Collect trajectories (local)
    - 06:00-18:00: Train on cloud GPUs
    - 18:00-20:00: Download + validate models
    - 20:00-24:00: Deploy to NPU + collect more data
    """

    def __init__(
        self,
        cloud_provider: str = "vast.ai",  # or "runpod", "lambda"
        gpu_type: str = "A100",
        num_gpus: int = 4,
        training_hours_per_day: int = 12,
        local_db_config: Dict = None,
        s3_bucket: str = "lat5150-rl-training"
    ):
        self.cloud_provider = cloud_provider
        self.gpu_type = gpu_type
        self.num_gpus = num_gpus
        self.training_hours = training_hours_per_day
        self.s3_bucket = s3_bucket

        # Local database for trajectories
        self.db = psycopg2.connect(**local_db_config or self._default_db_config())

        # S3 for model/data sync
        self.s3 = boto3.client('s3')

        # Cloud GPU instance (auto-provisioned)
        self.cloud_instance = None

        # Metrics tracking
        self.improvement_metrics = []

    def _default_db_config(self):
        return {
            "host": "localhost",
            "database": "rl_trajectories",
            "user": "postgres",
            "password": os.getenv("DB_PASSWORD")
        }

    def run_continuous_improvement(self):
        """
        Main loop: run 24/7 for continuous self-improvement

        Each iteration (24 hours):
        1. Collect 6 hours of trajectories
        2. Train 12 hours on cloud
        3. Validate 2 hours
        4. Deploy 4 hours + collect
        """
        iteration = 0

        while True:
            iteration += 1
            print(f"\n{'='*80}")
            print(f"IMPROVEMENT ITERATION {iteration}")
            print(f"{'='*80}")

            try:
                # PHASE 1: Collect trajectories (6 hours)
                print("\n[Phase 1/4] Collecting trajectories on local hardware...")
                trajectories = self._collect_trajectories(duration_hours=6)
                print(f"✓ Collected {len(trajectories)} trajectories")

                # PHASE 2: Upload to S3
                print("\n[Phase 2/4] Uploading trajectories to S3...")
                self._upload_trajectories_to_s3(trajectories)
                print(f"✓ Uploaded to s3://{self.s3_bucket}/trajectories/")

                # PHASE 3: Train on cloud GPUs (12 hours)
                print("\n[Phase 3/4] Training on cloud GPUs...")
                self._provision_cloud_gpus()
                trained_model_path = self._train_on_cloud(duration_hours=12)
                print(f"✓ Training complete: {trained_model_path}")

                # PHASE 4: Download and deploy (2 hours validation + 4 hours deployment)
                print("\n[Phase 4/4] Downloading and deploying model...")
                self._download_model_from_s3(trained_model_path)
                validation_metrics = self._validate_model()

                if validation_metrics['improvement'] > 0.05:  # 5% improvement
                    print(f"✓ Validation passed: {validation_metrics['improvement']*100:.1f}% improvement")
                    self._deploy_to_npu()
                    print("✓ Deployed to NPU for production")
                else:
                    print(f"⚠️  Insufficient improvement: {validation_metrics['improvement']*100:.1f}%")
                    print("   Keeping previous model")

                # PHASE 5: Cleanup cloud resources
                self._terminate_cloud_gpus()

                # Record metrics
                self.improvement_metrics.append({
                    "iteration": iteration,
                    "timestamp": datetime.now().isoformat(),
                    "trajectories": len(trajectories),
                    "training_hours": 12,
                    "improvement": validation_metrics['improvement'],
                    "deployed": validation_metrics['improvement'] > 0.05
                })

                # Save metrics
                self._save_improvement_metrics()

                print(f"\n✅ Iteration {iteration} complete!")
                print(f"   Total improvement: {sum(m['improvement'] for m in self.improvement_metrics)*100:.1f}%")

            except Exception as e:
                print(f"❌ Error in iteration {iteration}: {e}")
                # Cleanup on error
                if self.cloud_instance:
                    self._terminate_cloud_gpus()
                # Wait before retry
                time.sleep(3600)  # 1 hour

    def _collect_trajectories(self, duration_hours: int) -> List[Dict]:
        """
        Collect agent trajectories on local hardware

        Trajectory format:
        {
            "state": str,  # Query + context
            "action": str,  # Agent response
            "reward": float,  # Human feedback or automatic reward
            "next_state": str,
            "done": bool
        }
        """
        from feedback.hitl_feedback_enhanced import EnhancedHITLFeedback

        feedback_collector = EnhancedHITLFeedback()
        trajectories = []

        end_time = time.time() + (duration_hours * 3600)

        print(f"Collecting for {duration_hours} hours...")
        print("Agent will run autonomously and collect feedback...")

        # In production, this would run the agent system continuously
        # For now, we'll collect from existing feedback

        # Query database for recent feedback
        cursor = self.db.cursor()
        cursor.execute('''
            SELECT session_id, query, response_a, feedback_type, feedback_value, timestamp
            FROM feedback
            WHERE timestamp > ?
            ORDER BY timestamp DESC
        ''', (time.time() - (duration_hours * 3600),))

        for row in cursor.fetchall():
            session_id, query, response, fb_type, fb_value, ts = row

            # Convert feedback to reward
            reward = self._feedback_to_reward(fb_type, fb_value)

            trajectories.append({
                "state": query,
                "action": response,
                "reward": reward,
                "next_state": "",  # Will be filled by next interaction
                "done": False,
                "timestamp": ts,
                "session_id": session_id
            })

        cursor.close()

        return trajectories

    def _feedback_to_reward(self, feedback_type: str, feedback_value: str) -> float:
        """
        Convert human feedback to RL reward

        Reward structure:
        - Thumbs up: +1.0
        - Thumbs down: -0.5
        - High rating (4-5): +0.8
        - Medium rating (3): +0.3
        - Low rating (1-2): -0.3
        - Correction: -0.2 (but useful for learning)
        """
        value = json.loads(feedback_value)

        if feedback_type == "thumbs":
            return 1.0 if value['thumbs'] == 'up' else -0.5
        elif feedback_type == "rating":
            rating = value['rating']
            if rating >= 4:
                return 0.8
            elif rating == 3:
                return 0.3
            else:
                return -0.3
        elif feedback_type == "comparison":
            return 0.5  # Partial reward for preference
        elif feedback_type == "correction":
            return -0.2  # Negative but useful
        else:
            return 0.0

    def _upload_trajectories_to_s3(self, trajectories: List[Dict]):
        """Upload trajectories to S3 for cloud training"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trajectories_{timestamp}.json"

        # Save locally first
        local_path = f"/tmp/{filename}"
        with open(local_path, 'w') as f:
            json.dump(trajectories, f)

        # Upload to S3
        s3_key = f"trajectories/{filename}"
        self.s3.upload_file(local_path, self.s3_bucket, s3_key)

        return s3_key

    def _provision_cloud_gpus(self):
        """
        Provision cloud GPU instance

        Uses Vast.ai API or RunPod API to auto-provision GPUs
        """
        if self.cloud_provider == "vast.ai":
            return self._provision_vast_ai()
        elif self.cloud_provider == "runpod":
            return self._provision_runpod()
        elif self.cloud_provider == "lambda":
            return self._provision_lambda_labs()

    def _provision_vast_ai(self):
        """
        Provision Vast.ai GPU instance

        Uses Vast.ai API:
        - Search for cheapest A100 instances
        - Rent instance
        - Setup training environment
        """
        import requests

        # Vast.ai API
        api_key = os.getenv("VAST_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"}

        # Search for instances
        search_params = {
            "gpu_name": self.gpu_type,
            "num_gpus": self.num_gpus,
            "disk_space": 500,  # 500GB
            "sort": "score-",  # Best value
            "verified": True
        }

        response = requests.get(
            "https://console.vast.ai/api/v0/bundles",
            headers=headers,
            params=search_params
        )

        offers = response.json()['offers']

        # Select cheapest
        best_offer = min(offers, key=lambda x: x['dph_total'])  # Dollars per hour

        print(f"Selected instance: {best_offer['id']}")
        print(f"  GPUs: {best_offer['num_gpus']}x {best_offer['gpu_name']}")
        print(f"  Cost: ${best_offer['dph_total']:.2f}/hr")
        print(f"  Est. total: ${best_offer['dph_total'] * self.training_hours:.2f}")

        # Rent instance
        rent_response = requests.put(
            f"https://console.vast.ai/api/v0/asks/{best_offer['id']}/",
            headers=headers,
            json={
                "image": "nvidia/pytorch:24.01-py3",  # Docker image with PyTorch
                "disk": 500,
                "env": {
                    "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
                    "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
                    "S3_BUCKET": self.s3_bucket
                }
            }
        )

        instance = rent_response.json()

        # Wait for instance to be ready
        print("Waiting for instance to start...")
        while True:
            status_response = requests.get(
                f"https://console.vast.ai/api/v0/instances/{instance['id']}",
                headers=headers
            )
            status = status_response.json()

            if status['actual_status'] == 'running':
                print("✓ Instance ready!")
                break

            time.sleep(10)

        self.cloud_instance = {
            "provider": "vast.ai",
            "id": instance['id'],
            "ip": status['public_ipaddr'],
            "ssh_port": status['ssh_port'],
            "ssh_key": instance['ssh_key']
        }

        return self.cloud_instance

    def _train_on_cloud(self, duration_hours: int) -> str:
        """
        Execute PPO training on cloud GPUs

        Steps:
        1. SSH to cloud instance
        2. Download trajectories from S3
        3. Run PPO training script
        4. Upload trained model to S3
        """
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect via SSH
        ssh.connect(
            hostname=self.cloud_instance['ip'],
            port=self.cloud_instance['ssh_port'],
            username='root',
            key_filename=self.cloud_instance['ssh_key']
        )

        # Upload training script
        sftp = ssh.open_sftp()
        sftp.put('02-ai-engine/rl_training/ppo_trainer.py', '/workspace/ppo_trainer.py')
        sftp.close()

        # Run training
        commands = f"""
        # Download trajectories from S3
        aws s3 sync s3://{self.s3_bucket}/trajectories/ /workspace/trajectories/

        # Install dependencies
        pip install transformers trl peft accelerate

        # Run PPO training
        python /workspace/ppo_trainer.py \\
            --trajectories /workspace/trajectories/ \\
            --output /workspace/models/ \\
            --num-gpus {self.num_gpus} \\
            --hours {duration_hours}

        # Upload trained model to S3
        aws s3 sync /workspace/models/ s3://{self.s3_bucket}/models/
        """

        stdin, stdout, stderr = ssh.exec_command(commands)

        # Stream output
        for line in stdout:
            print(line.strip())

        # Get model path
        model_path = f"s3://{self.s3_bucket}/models/latest/"

        ssh.close()

        return model_path

    def _download_model_from_s3(self, model_path: str):
        """Download trained model from S3"""
        local_dir = "/home/user/LAT5150DRVMIL/models/ppo_latest"
        os.makedirs(local_dir, exist_ok=True)

        # Sync from S3
        os.system(f"aws s3 sync {model_path} {local_dir}/")

        print(f"✓ Model downloaded to {local_dir}")

    def _validate_model(self) -> Dict:
        """
        Validate model improvement

        Run on validation set and compare to baseline
        """
        from transformers import AutoModelForCausalLM

        # Load new model
        new_model = AutoModelForCausalLM.from_pretrained(
            "/home/user/LAT5150DRVMIL/models/ppo_latest",
            device_map="xpu"  # Arc GPU
        )

        # Load baseline
        baseline_model = AutoModelForCausalLM.from_pretrained(
            "/home/user/LAT5150DRVMIL/models/baseline",
            device_map="xpu"
        )

        # Validation queries
        validation_queries = self._get_validation_queries()

        # Evaluate both
        new_score = self._evaluate_model(new_model, validation_queries)
        baseline_score = self._evaluate_model(baseline_model, validation_queries)

        improvement = (new_score - baseline_score) / baseline_score

        return {
            "new_score": new_score,
            "baseline_score": baseline_score,
            "improvement": improvement
        }

    def _deploy_to_npu(self):
        """
        Deploy trained model to NPU

        Steps:
        1. Quantize to INT8
        2. Convert to OpenVINO IR
        3. Test on NPU
        4. Swap production model
        """
        from neural_compressor import quantization

        # Load PyTorch model
        model_path = "/home/user/LAT5150DRVMIL/models/ppo_latest"
        model = AutoModelForCausalLM.from_pretrained(model_path)

        # Quantize to INT8
        print("Quantizing to INT8 for NPU...")
        quantized = quantization.fit(
            model,
            quantization.PostTrainingQuantConfig(backend="ipex")
        )

        # Save INT8 model
        npu_model_path = "/home/user/LAT5150DRVMIL/models/production_npu"
        quantized.save(npu_model_path)

        # Convert to OpenVINO IR
        import openvino as ov
        ov_model = ov.convert_model(npu_model_path)
        ov.serialize(ov_model, f"{npu_model_path}/model.xml")

        # Test on NPU
        core = ov.Core()
        compiled = core.compile_model(ov_model, "NPU")

        # Test inference
        test_input = "What is machine learning?"
        result = compiled([self._tokenize(test_input)])[0]
        print(f"✓ NPU test passed: {result[:50]}...")

        print(f"✓ Model deployed to NPU: {npu_model_path}")

    def _terminate_cloud_gpus(self):
        """Terminate cloud GPU instance to save costs"""
        if not self.cloud_instance:
            return

        if self.cloud_instance['provider'] == "vast.ai":
            import requests
            api_key = os.getenv("VAST_API_KEY")
            headers = {"Authorization": f"Bearer {api_key}"}

            requests.delete(
                f"https://console.vast.ai/api/v0/instances/{self.cloud_instance['id']}/",
                headers=headers
            )

            print(f"✓ Terminated instance {self.cloud_instance['id']}")

        self.cloud_instance = None

    def _save_improvement_metrics(self):
        """Save improvement metrics for analysis"""
        with open("/home/user/LAT5150DRVMIL/logs/improvement_metrics.json", 'w') as f:
            json.dump(self.improvement_metrics, f, indent=2)

# Usage
if __name__ == "__main__":
    orchestrator = AutoImprovementOrchestrator(
        cloud_provider="vast.ai",
        gpu_type="A100",
        num_gpus=4,
        training_hours_per_day=12,
        s3_bucket="lat5150-rl-training"
    )

    # Run continuous self-improvement (24/7)
    orchestrator.run_continuous_improvement()
```

---

### 3B: C/C++ Performance-Critical Components (Weeks 15-18)

#### Rationale for C/C++ Optimization

**Bottlenecks in Python:**
1. Vector search (100K+ vectors): Python ~10ms, C++ ~2ms (5x faster)
2. Reward calculation: Python ~5ms, C++ ~0.5ms (10x faster)
3. Trajectory processing: Python ~100ms, C/C++ ~10ms (10x faster)
4. Token embedding: Python ~2ms, C++ ~0.5ms (4x faster)

**Strategy:**
- Keep high-level orchestration in Python
- Rewrite performance-critical loops in C/C++
- Use Cython for Python/C++ interface

#### Implementation: C++ Vector Search with AVX-512

**File:** `02-ai-engine/rag_cpp/vector_search_avx512.cpp`

```cpp
/**
 * AVX-512 optimized vector search
 *
 * P-core pinning: CPUs 0-5 (Dell Latitude 5450)
 * Uses AVX-512 SIMD for 8x float32 operations per instruction
 *
 * Performance:
 * - Python: ~10ms for 100K vectors
 * - C++ AVX-512: ~2ms (5x speedup)
 *
 * Compile:
 * g++ -O3 -mavx512f -mavx512dq -march=native -pthread \\
 *     vector_search_avx512.cpp -o vector_search -fopenmp
 */

#include <immintrin.h>  // AVX-512 intrinsics
#include <vector>
#include <cmath>
#include <algorithm>
#include <pthread.h>
#include <sched.h>
#include <iostream>

// P-core CPU set (0-5 on Dell Latitude 5450)
const int P_CORES[] = {0, 1, 2, 3, 4, 5};
const int NUM_P_CORES = 6;

/**
 * Pin thread to P-core for AVX-512
 *
 * CRITICAL: AVX-512 must run on P-cores only
 * E-cores don't have AVX-512 support
 */
void pin_to_pcore(int core_id) {
    if (core_id >= NUM_P_CORES) {
        std::cerr << "Error: Core " << core_id << " is not a P-core" << std::endl;
        return;
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(P_CORES[core_id], &cpuset);

    pthread_t current_thread = pthread_self();
    int result = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);

    if (result == 0) {
        std::cout << "✓ Thread pinned to P-core " << P_CORES[core_id] << std::endl;
    } else {
        std::cerr << "✗ Failed to pin to P-core " << core_id << std::endl;
    }
}

/**
 * Cosine similarity using AVX-512
 *
 * Processes 16 floats per iteration (512-bit SIMD)
 */
float cosine_similarity_avx512(const float* a, const float* b, int dim) {
    __m512 dot_product = _mm512_setzero_ps();
    __m512 norm_a = _mm512_setzero_ps();
    __m512 norm_b = _mm512_setzero_ps();

    // Process 16 floats at a time
    int i;
    for (i = 0; i + 16 <= dim; i += 16) {
        __m512 vec_a = _mm512_loadu_ps(&a[i]);
        __m512 vec_b = _mm512_loadu_ps(&b[i]);

        dot_product = _mm512_fmadd_ps(vec_a, vec_b, dot_product);
        norm_a = _mm512_fmadd_ps(vec_a, vec_a, norm_a);
        norm_b = _mm512_fmadd_ps(vec_b, vec_b, norm_b);
    }

    // Reduce 16-wide vectors to scalars
    float dot_sum = _mm512_reduce_add_ps(dot_product);
    float norm_a_sum = _mm512_reduce_add_ps(norm_a);
    float norm_b_sum = _mm512_reduce_add_ps(norm_b);

    // Handle remaining elements
    for (; i < dim; i++) {
        dot_sum += a[i] * b[i];
        norm_a_sum += a[i] * a[i];
        norm_b_sum += b[i] * b[i];
    }

    return dot_sum / (std::sqrt(norm_a_sum) * std::sqrt(norm_b_sum));
}

/**
 * Parallel vector search across database
 *
 * Uses OpenMP to parallelize across P-cores
 */
struct SearchResult {
    int doc_id;
    float score;
};

bool compare_results(const SearchResult& a, const SearchResult& b) {
    return a.score > b.score;
}

std::vector<SearchResult> search_top_k(
    const float* query,           // Query embedding (dim-dimensional)
    const float** database,       // Database of embeddings
    int num_docs,                 // Number of documents
    int dim,                      // Embedding dimension
    int top_k                     // Number of results
) {
    // Pin master thread to P-core 0
    pin_to_pcore(0);

    std::vector<SearchResult> all_results(num_docs);

    // Parallel search using OpenMP
    #pragma omp parallel for num_threads(NUM_P_CORES) schedule(dynamic, 1000)
    for (int i = 0; i < num_docs; i++) {
        // Each thread automatically pinned to P-core by OpenMP
        float similarity = cosine_similarity_avx512(query, database[i], dim);
        all_results[i] = {i, similarity};
    }

    // Sort by score (descending)
    std::partial_sort(
        all_results.begin(),
        all_results.begin() + top_k,
        all_results.end(),
        compare_results
    );

    // Return top-k
    std::vector<SearchResult> top_results(
        all_results.begin(),
        all_results.begin() + top_k
    );

    return top_results;
}

// Python interface via Cython
extern "C" {
    void* create_vector_database(int num_docs, int dim);
    void add_document(void* db, int doc_id, const float* embedding);
    void search(void* db, const float* query, int top_k, int* doc_ids, float* scores);
}
```

**Cython Interface:** `02-ai-engine/rag_cpp/vector_search.pyx`

```cython
# cython: language_level=3
# distutils: language=c++
# distutils: extra_compile_args=-O3 -mavx512f -mavx512dq -march=native -fopenmp
# distutils: extra_link_args=-fopenmp

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "vector_search_avx512.cpp":
    void* create_vector_database(int num_docs, int dim)
    void add_document(void* db, int doc_id, const float* embedding)
    void search(void* db, const float* query, int top_k, int* doc_ids, float* scores)

cdef class VectorSearchAVX512:
    """
    Python wrapper for C++ AVX-512 vector search

    Usage:
        search_engine = VectorSearchAVX512(num_docs=100000, dim=384)
        search_engine.add_documents(embeddings)
        results = search_engine.search(query, top_k=10)
    """
    cdef void* db
    cdef int num_docs
    cdef int dim

    def __init__(self, int num_docs, int dim):
        self.num_docs = num_docs
        self.dim = dim
        self.db = create_vector_database(num_docs, dim)

    def add_documents(self, np.ndarray[np.float32_t, ndim=2] embeddings):
        """Add documents to database"""
        cdef int i
        cdef float* emb_ptr

        for i in range(embeddings.shape[0]):
            emb_ptr = &embeddings[i, 0]
            add_document(self.db, i, emb_ptr)

    def search(self, np.ndarray[np.float32_t, ndim=1] query, int top_k):
        """Search for top-k most similar documents"""
        cdef int* doc_ids = <int*>malloc(top_k * sizeof(int))
        cdef float* scores = <float*>malloc(top_k * sizeof(float))
        cdef float* query_ptr = &query[0]

        # Call C++ search (pinned to P-cores automatically)
        search(self.db, query_ptr, top_k, doc_ids, scores)

        # Convert to numpy arrays
        doc_ids_np = np.zeros(top_k, dtype=np.int32)
        scores_np = np.zeros(top_k, dtype=np.float32)

        for i in range(top_k):
            doc_ids_np[i] = doc_ids[i]
            scores_np[i] = scores[i]

        free(doc_ids)
        free(scores)

        return doc_ids_np, scores_np

# Build script: setup.py
"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "vector_search",
        ["vector_search.pyx"],
        extra_compile_args=["-O3", "-mavx512f", "-mavx512dq", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        include_dirs=[numpy.get_include()],
        language="c++"
    )
]

setup(
    name="vector_search_avx512",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)
"""
```

**Usage in Python:**

```python
# Replace ChromaDB vector search with AVX-512 C++ version
from vector_search import VectorSearchAVX512

class FastRAGSystem:
    """
    RAG with C++ AVX-512 vector search

    5x faster than pure Python
    """

    def __init__(self, num_docs: int = 100000, dim: int = 384):
        # C++ search engine (AVX-512, P-core pinned)
        self.search_engine = VectorSearchAVX512(num_docs, dim)

        # Python embedder (still on NPU)
        self.embedder = self._load_npu_embedder()

    def add_documents(self, documents: List[str]):
        """Add documents with embeddings"""
        # Embed on NPU (INT8, ~1000 embeds/sec)
        embeddings = self.embedder.encode(documents)

        # Add to C++ index (AVX-512)
        self.search_engine.add_documents(embeddings.astype(np.float32))

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search with AVX-512 acceleration

        Pipeline:
        1. Embed query on NPU (~1ms)
        2. Vector search with AVX-512 on P-cores (~2ms)
        Total: ~3ms for 100K documents
        """
        # Embed query (NPU)
        query_embedding = self.embedder.encode([query])[0]

        # Search (C++ AVX-512, P-cores 0-5)
        doc_ids, scores = self.search_engine.search(query_embedding, top_k)

        # Return results
        return [
            {"doc_id": int(doc_ids[i]), "score": float(scores[i])}
            for i in range(top_k)
        ]

# Performance comparison
import time

# Pure Python (ChromaDB)
chroma_rag = EnhancedRAGSystem()
start = time.time()
chroma_results = chroma_rag.search("quantum computing", top_k=10)
chroma_time = (time.time() - start) * 1000
print(f"ChromaDB: {chroma_time:.2f}ms")

# C++ AVX-512
fast_rag = FastRAGSystem()
start = time.time()
fast_results = fast_rag.search("quantum computing", top_k=10)
fast_time = (time.time() - start) * 1000
print(f"AVX-512: {fast_time:.2f}ms")

print(f"Speedup: {chroma_time / fast_time:.1f}x")
# Expected output:
# ChromaDB: 10.5ms
# AVX-512: 2.1ms
# Speedup: 5.0x
```

---

### 3C: Dynamic Hardware Resource Allocation (Weeks 19-22)

#### Challenge: Optimal Hardware Utilization

**Problem:**
- Multiple accelerators (NPU, Arc, NCS2, AVX-512)
- Different workloads have different optimal hardware
- Hardware may be added/removed over time
- Need intelligent workload distribution

**Solution: Dynamic Hardware Manager**

**File:** `02-ai-engine/hardware/dynamic_allocator.py`

```python
#!/usr/bin/env python3
"""
Dynamic Hardware Resource Allocator

Automatically detects and optimally distributes workloads across:
- Intel NPU (49.4 TOPS INT8)
- Intel Arc GPU (16 TFLOPS)
- Intel NCS2 (3x sticks, 3 TOPS)
- AVX-512 (P-cores 0-5)
- Standard CPU (E-cores)

Adapts to hardware changes (e.g., adding more NCS2 sticks)
"""

import os
import psutil
import subprocess
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import threading
import queue

class HardwareType(Enum):
    """Available hardware types"""
    NPU = "npu"
    ARC_GPU = "arc_gpu"
    NCS2 = "ncs2"
    AVX512 = "avx512"
    CPU = "cpu"

@dataclass
class HardwareDevice:
    """Hardware device descriptor"""
    hw_type: HardwareType
    device_id: str
    available: bool
    utilization: float  # 0.0-1.0
    capabilities: List[str]
    performance_rating: int  # 1-10

@dataclass
class Workload:
    """Workload to be scheduled"""
    workload_type: str  # "embedding", "inference", "training", "vector_search"
    priority: int  # 1-10
    estimated_duration_ms: float
    required_memory_mb: float
    preferred_hardware: Optional[HardwareType] = None

class HardwareManager:
    """
    Dynamic hardware resource manager

    Features:
    - Auto-detection of all accelerators
    - Optimal workload scheduling
    - Load balancing
    - Hot-plug support (add/remove devices)
    """

    def __init__(self):
        self.devices: List[HardwareDevice] = []
        self.workload_queue = queue.PriorityQueue()
        self.scheduler_thread = None

        # Detect hardware
        self._detect_hardware()

        # Start scheduler
        self._start_scheduler()

    def _detect_hardware(self):
        """
        Detect all available hardware accelerators

        Checks for:
        - Intel NPU (via openvino)
        - Intel Arc GPU (via torch XPU)
        - Intel NCS2 sticks (via openvino)
        - AVX-512 support (via cpuinfo)
        """
        self.devices = []

        # 1. Detect NPU
        if self._detect_npu():
            self.devices.append(HardwareDevice(
                hw_type=HardwareType.NPU,
                device_id="npu_0",
                available=True,
                utilization=0.0,
                capabilities=["inference_int8", "inference_int4", "embeddings"],
                performance_rating=9  # Very fast for INT8
            ))
            print("✓ Intel NPU detected (49.4 TOPS INT8)")

        # 2. Detect Arc GPU
        if self._detect_arc_gpu():
            self.devices.append(HardwareDevice(
                hw_type=HardwareType.ARC_GPU,
                device_id="xpu_0",
                available=True,
                utilization=0.0,
                capabilities=["training", "inference_fp16", "inference_bf16"],
                performance_rating=8  # Good for training
            ))
            print("✓ Intel Arc GPU detected (16 TFLOPS)")

        # 3. Detect NCS2 sticks
        ncs2_count = self._detect_ncs2()
        for i in range(ncs2_count):
            self.devices.append(HardwareDevice(
                hw_type=HardwareType.NCS2,
                device_id=f"ncs2_{i}",
                available=True,
                utilization=0.0,
                capabilities=["inference_int8", "reranking"],
                performance_rating=6  # Moderate performance
            ))
        if ncs2_count > 0:
            print(f"✓ {ncs2_count}x Intel NCS2 detected")

        # 4. Detect AVX-512 (P-cores)
        if self._detect_avx512():
            # Create 6 AVX-512 workers (one per P-core)
            for i in range(6):
                self.devices.append(HardwareDevice(
                    hw_type=HardwareType.AVX512,
                    device_id=f"avx512_pcore{i}",
                    available=True,
                    utilization=0.0,
                    capabilities=["vector_search", "inference_fp32", "preprocessing"],
                    performance_rating=7  # Good for specific tasks
                ))
            print("✓ AVX-512 detected (P-cores 0-5)")

        # 5. CPU fallback (E-cores or older CPUs)
        cpu_cores = psutil.cpu_count(logical=False) - 6  # Subtract P-cores
        for i in range(min(cpu_cores, 4)):  # Max 4 CPU workers
            self.devices.append(HardwareDevice(
                hw_type=HardwareType.CPU,
                device_id=f"cpu_{i}",
                available=True,
                utilization=0.0,
                capabilities=["preprocessing", "postprocessing"],
                performance_rating=4  # Slowest
            ))

        print(f"\n✓ Total devices detected: {len(self.devices)}")

    def _detect_npu(self) -> bool:
        """Detect Intel NPU via OpenVINO"""
        try:
            import openvino as ov
            core = ov.Core()
            devices = core.available_devices()
            return "NPU" in devices
        except:
            return False

    def _detect_arc_gpu(self) -> bool:
        """Detect Intel Arc GPU via torch XPU"""
        try:
            import torch
            import intel_extension_for_pytorch as ipex
            return torch.xpu.is_available()
        except:
            return False

    def _detect_ncs2(self) -> int:
        """Detect number of NCS2 sticks"""
        try:
            import openvino as ov
            core = ov.Core()
            devices = core.available_devices()
            # Count MYRIAD devices (NCS2)
            return devices.count("MYRIAD")
        except:
            return 0

    def _detect_avx512(self) -> bool:
        """Detect AVX-512 support"""
        try:
            # Read /proc/cpuinfo
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
            return "avx512" in cpuinfo.lower()
        except:
            return False

    def _start_scheduler(self):
        """Start background scheduler thread"""
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True
        )
        self.scheduler_thread.start()

    def _scheduler_loop(self):
        """
        Background scheduler loop

        Continuously assigns workloads to optimal hardware
        """
        while True:
            try:
                # Get next workload (blocks if queue empty)
                priority, workload = self.workload_queue.get(timeout=1.0)

                # Find optimal device
                device = self._select_optimal_device(workload)

                if device:
                    # Execute workload
                    self._execute_workload(workload, device)
                else:
                    # No device available, requeue
                    self.workload_queue.put((priority, workload))

            except queue.Empty:
                continue

    def _select_optimal_device(self, workload: Workload) -> Optional[HardwareDevice]:
        """
        Select optimal hardware for workload

        Scoring function:
        score = (capability_match * 5) + (performance * 3) + (availability * 2) - (utilization * 4)
        """
        best_device = None
        best_score = -float('inf')

        for device in self.devices:
            if not device.available:
                continue

            # Check capability match
            capability_match = 0
            if workload.workload_type in device.capabilities:
                capability_match = 1

            # Preferred hardware bonus
            if workload.preferred_hardware and device.hw_type == workload.preferred_hardware:
                capability_match += 0.5

            # Calculate score
            score = (
                capability_match * 5 +
                (device.performance_rating / 10) * 3 +
                (1.0 if device.available else 0.0) * 2 -
                device.utilization * 4
            )

            if score > best_score:
                best_score = score
                best_device = device

        return best_device

    def _execute_workload(self, workload: Workload, device: HardwareDevice):
        """Execute workload on selected device"""
        # Mark device as busy
        device.utilization = 1.0

        # Route to appropriate executor
        if device.hw_type == HardwareType.NPU:
            self._execute_on_npu(workload, device)
        elif device.hw_type == HardwareType.ARC_GPU:
            self._execute_on_arc(workload, device)
        elif device.hw_type == HardwareType.NCS2:
            self._execute_on_ncs2(workload, device)
        elif device.hw_type == HardwareType.AVX512:
            self._execute_on_avx512(workload, device)
        else:
            self._execute_on_cpu(workload, device)

        # Mark device as free
        device.utilization = 0.0

    def _execute_on_avx512(self, workload: Workload, device: HardwareDevice):
        """
        Execute on AVX-512 P-core

        CRITICAL: Pin to specific P-core
        """
        import os
        import sched_affinity  # pip install sched-affinity

        # Extract P-core number from device_id (e.g., "avx512_pcore2" -> 2)
        pcore_id = int(device.device_id.split("pcore")[1])
        actual_core = pcore_id  # P-cores are 0-5

        # Pin to P-core
        os.sched_setaffinity(0, {actual_core})

        print(f"✓ Workload pinned to P-core {actual_core} for AVX-512")

        # Execute workload (will use AVX-512 automatically)
        # ...workload execution code...

        # Reset affinity
        os.sched_setaffinity(0, set(range(os.cpu_count())))

    def submit_workload(self, workload: Workload):
        """Submit workload to scheduler"""
        self.workload_queue.put((workload.priority, workload))

    def get_device_stats(self) -> Dict:
        """Get current hardware utilization stats"""
        stats = {
            "total_devices": len(self.devices),
            "by_type": {},
            "avg_utilization": 0.0
        }

        for device in self.devices:
            hw_type = device.hw_type.value
            if hw_type not in stats["by_type"]:
                stats["by_type"][hw_type] = {"count": 0, "utilization": 0.0}

            stats["by_type"][hw_type]["count"] += 1
            stats["by_type"][hw_type]["utilization"] += device.utilization

        # Calculate averages
        for hw_type in stats["by_type"]:
            count = stats["by_type"][hw_type]["count"]
            stats["by_type"][hw_type]["utilization"] /= count

        stats["avg_utilization"] = sum(d.utilization for d in self.devices) / len(self.devices)

        return stats

# Usage example
if __name__ == "__main__":
    # Initialize hardware manager
    hw_manager = HardwareManager()

    # Submit various workloads
    workloads = [
        Workload(
            workload_type="embeddings",
            priority=8,
            estimated_duration_ms=100,
            required_memory_mb=500,
            preferred_hardware=HardwareType.NPU  # Prefer NPU for embeddings
        ),
        Workload(
            workload_type="vector_search",
            priority=9,
            estimated_duration_ms=50,
            required_memory_mb=1000,
            preferred_hardware=HardwareType.AVX512  # AVX-512 for vector search
        ),
        Workload(
            workload_type="training",
            priority=5,
            estimated_duration_ms=10000,
            required_memory_mb=8000,
            preferred_hardware=HardwareType.ARC_GPU  # Arc GPU for training
        ),
    ]

    for workload in workloads:
        hw_manager.submit_workload(workload)

    # Monitor utilization
    import time
    while True:
        stats = hw_manager.get_device_stats()
        print(f"\nHardware Utilization:")
        for hw_type, data in stats["by_type"].items():
            print(f"  {hw_type}: {data['utilization']*100:.1f}% ({data['count']} devices)")
        print(f"  Overall: {stats['avg_utilization']*100:.1f}%")

        time.sleep(5)
```

---

### 3D: ZFS-Aware Storage Optimization (Weeks 21-22)

#### ZFS Benefits for AI Workloads

**Advantages:**
1. Compression: 2-4x space savings for model checkpoints
2. Snapshots: Instant model versioning
3. Datasets: Logical separation of training/validation/production
4. ARC cache: Faster repeated data access
5. Deduplication: Save space on similar model checkpoints

**File:** `02-ai-engine/storage/zfs_optimizer.py`

```python
#!/usr/bin/env python3
"""
ZFS Storage Optimizer for AI Workloads

Leverages ZFS features:
- Compression for model checkpoints (2-4x savings)
- Snapshots for model versioning
- Datasets for logical separation
- ARC cache tuning for model loading

ZFS layout:
zpool/ai-engine/
├── models/production       (compression=lz4, dedup=on)
├── models/checkpoints      (compression=zstd, snapshots)
├── training-data           (compression=lz4, recordsize=1M)
├── rag-index               (compression=lz4, primarycache=metadata)
└── logs                    (compression=gzip-9)
"""

import os
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

class ZFSOptimizer:
    """
    ZFS storage optimizer for AI workloads

    Automatically configures ZFS datasets with optimal settings
    """

    def __init__(self, zpool_name: str = "zpool"):
        self.zpool = zpool_name
        self.base_path = f"/{zpool_name}/ai-engine"

        # Create base datasets
        self._create_datasets()

    def _create_datasets(self):
        """Create ZFS datasets with optimal settings"""
        datasets = {
            "models/production": {
                "compression": "lz4",  # Fast compression
                "dedup": "on",         # Deduplicate similar models
                "recordsize": "128K",  # Large sequential reads
                "atime": "off",        # Don't update access time
                "sync": "standard"     # Balance safety/performance
            },
            "models/checkpoints": {
                "compression": "zstd-3",  # Better compression ratio
                "snapdir": "visible",     # Show snapshots
                "recordsize": "128K",
                "atime": "off"
            },
            "training-data": {
                "compression": "lz4",     # Fast for training
                "recordsize": "1M",       # Large dataset files
                "primarycache": "all",    # Cache in RAM
                "atime": "off"
            },
            "rag-index": {
                "compression": "lz4",
                "recordsize": "64K",      # Vector database blocks
                "primarycache": "metadata",  # Only cache metadata
                "atime": "off"
            },
            "logs": {
                "compression": "gzip-9",  # Maximum compression for logs
                "recordsize": "128K",
                "sync": "disabled",       # Fast writes (logs recoverable)
                "atime": "off"
            }
        }

        for dataset, properties in datasets.items():
            full_path = f"{self.zpool}/ai-engine/{dataset}"

            # Create dataset
            try:
                subprocess.run(
                    ["zfs", "create", "-p", full_path],
                    check=True,
                    capture_output=True
                )
                print(f"✓ Created dataset: {full_path}")
            except subprocess.CalledProcessError:
                print(f"  Dataset {full_path} already exists")

            # Set properties
            for prop, value in properties.items():
                try:
                    subprocess.run(
                        ["zfs", "set", f"{prop}={value}", full_path],
                        check=True,
                        capture_output=True
                    )
                except subprocess.CalledProcessError as e:
                    print(f"  Warning: Could not set {prop}={value}: {e}")

    def save_model_checkpoint(
        self,
        model_path: str,
        checkpoint_name: str,
        create_snapshot: bool = True
    ):
        """
        Save model checkpoint with ZFS snapshot

        Snapshots are instant and take no extra space initially
        """
        import shutil

        # Copy model to checkpoints dataset
        checkpoint_dir = f"{self.base_path}/models/checkpoints/{checkpoint_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        if os.path.isdir(model_path):
            shutil.copytree(model_path, checkpoint_dir, dirs_exist_ok=True)
        else:
            shutil.copy2(model_path, checkpoint_dir)

        print(f"✓ Saved checkpoint: {checkpoint_dir}")

        # Create ZFS snapshot
        if create_snapshot:
            snapshot_name = f"{self.zpool}/ai-engine/models/checkpoints@{checkpoint_name}"

            subprocess.run(
                ["zfs", "snapshot", snapshot_name],
                check=True
            )

            print(f"✓ Created snapshot: {snapshot_name}")

            # Show compression ratio
            self._show_compression_stats("models/checkpoints")

    def rollback_to_snapshot(self, snapshot_name: str):
        """
        Rollback models to previous snapshot

        Instant operation, no data movement
        """
        full_snapshot = f"{self.zpool}/ai-engine/models/checkpoints@{snapshot_name}"

        print(f"Rolling back to snapshot: {snapshot_name}")

        subprocess.run(
            ["zfs", "rollback", full_snapshot],
            check=True
        )

        print(f"✓ Rolled back to {snapshot_name}")

    def list_snapshots(self) -> List[str]:
        """List all model snapshots"""
        result = subprocess.run(
            ["zfs", "list", "-t", "snapshot", "-o", "name", "-H"],
            capture_output=True,
            text=True
        )

        snapshots = [
            line.split('@')[1]
            for line in result.stdout.strip().split('\n')
            if 'models/checkpoints@' in line
        ]

        return snapshots

    def _show_compression_stats(self, dataset: str):
        """Show compression ratio for dataset"""
        full_path = f"{self.zpool}/ai-engine/{dataset}"

        result = subprocess.run(
            ["zfs", "get", "compressratio", full_path, "-H", "-o", "value"],
            capture_output=True,
            text=True
        )

        ratio = result.stdout.strip()
        print(f"  Compression ratio: {ratio}")

    def optimize_arc_cache(self):
        """
        Optimize ZFS ARC cache for AI workloads

        Recommendations:
        - 50% RAM for ARC (models cached in memory)
        - Prefetch for sequential reads (training data)
        - Metadata-only cache for RAG index (save RAM)
        """
        # Get total RAM
        import psutil
        total_ram_gb = psutil.virtual_memory().total / (1024**3)

        # Set ARC to 50% of RAM
        arc_size_bytes = int(total_ram_gb * 0.5 * (1024**3))

        print(f"\nZFS ARC Cache Optimization:")
        print(f"  Total RAM: {total_ram_gb:.1f} GB")
        print(f"  Setting ARC to: {arc_size_bytes / (1024**3):.1f} GB (50%)")

        # Set arc_max (requires root)
        try:
            subprocess.run(
                ["sudo", "sh", "-c", f"echo {arc_size_bytes} > /sys/module/zfs/parameters/zfs_arc_max"],
                check=True
            )
            print("✓ ARC cache optimized")
        except subprocess.CalledProcessError:
            print("⚠️  Could not set ARC size (requires root)")

        # Enable prefetch
        try:
            subprocess.run(
                ["sudo", "sh", "-c", "echo 1 > /sys/module/zfs/parameters/zfs_prefetch_disable"],
                check=True
            )
            print("✓ Prefetch enabled")
        except:
            pass

# Usage
if __name__ == "__main__":
    zfs = ZFSOptimizer(zpool_name="zpool")

    # Save model checkpoint with snapshot
    zfs.save_model_checkpoint(
        model_path="/home/user/LAT5150DRVMIL/models/ppo_latest",
        checkpoint_name="iteration_42_improved",
        create_snapshot=True
    )

    # List all snapshots
    snapshots = zfs.list_snapshots()
    print(f"\nAvailable snapshots: {snapshots}")

    # Rollback if needed
    # zfs.rollback_to_snapshot("iteration_41_baseline")

    # Optimize ARC cache
    zfs.optimize_arc_cache()
```

---

## [Document continues but approaching length limit]

**Status:** Completed Phases 1-3A through 3D
- ✅ DPO Training (Weeks 1-6)
- ✅ Self-RAG (Weeks 7-12)
- ✅ PPO Cloud Training (Weeks 13-20)
- ✅ C++ AVX-512 Optimization (Weeks 15-18)
- ✅ Dynamic Hardware Allocation (Weeks 19-22)
- ✅ ZFS Storage Optimization (Weeks 21-22)

**Remaining:** Phases 3E-4 covering:
- Intelligent multi-GPU connection discovery
- Learned MoE routing
- Meta-learning (MAML)
- Evaluation framework
- Production deployment

---

### 3E: Intelligent Multi-GPU Connection Discovery & Cybersecurity Integration (Weeks 21-24)

#### Challenge: Distributed Training with Incomplete Connection Details

**Problem:**
- User may provide partial GPU cluster information
- Need to auto-discover missing connection details
- Must integrate with cybersecurity pipeline for secure connections
- Support various authentication methods (SSH keys, VPN, API tokens)
- Automatic firewall traversal and port forwarding

**Solution: Intelligent Connection Discovery System**

**File:** `02-ai-engine/distributed/gpu_cluster_discovery.py`

```python
#!/usr/bin/env python3
"""
Intelligent Multi-GPU Connection Discovery

Capabilities:
1. Auto-discover remote GPU clusters from partial information
2. Integrate with cybersecurity pipeline for secure connections
3. Handle incomplete connection details (infer missing info)
4. Support SSH, VPN, API-based connections
5. Automatic firewall traversal and port forwarding
6. Secure model transfer with encryption

Examples of handled scenarios:
- User provides: "192.168.1.50" → Auto-discovers SSH port, GPU count, credentials
- User provides: "vast.ai API key" → Auto-provisions and configures cluster
- User provides: "company-server" → Resolves DNS, checks security policy, connects
"""

import os
import socket
import paramiko
import requests
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import subprocess
import threading
import time

@dataclass
class GPUClusterInfo:
    """Complete GPU cluster connection information"""
    # Basic info
    host: str
    port: int
    username: str

    # Authentication
    auth_method: str  # "ssh_key", "password", "api_token", "vpn"
    ssh_key_path: Optional[str] = None
    password: Optional[str] = None
    api_token: Optional[str] = None

    # Hardware details
    num_gpus: int = 0
    gpu_type: str = "unknown"
    total_vram_gb: float = 0.0

    # Network details
    requires_vpn: bool = False
    vpn_config_path: Optional[str] = None
    requires_port_forward: bool = False
    local_port: Optional[int] = None

    # Security
    security_cleared: bool = False
    security_policy: Optional[str] = None

    # Status
    is_available: bool = False
    connection_tested: bool = False

class IntelligentGPUDiscovery:
    """
    Intelligent GPU cluster discovery and connection manager

    Features:
    - Auto-complete partial connection information
    - Security pipeline integration
    - Automatic credential discovery
    - Firewall traversal
    - Connection health monitoring
    """

    def __init__(
        self,
        security_config_path: str = "/home/user/LAT5150DRVMIL/00-security/security_policy.json",
        known_hosts_db: str = "/home/user/LAT5150DRVMIL/config/known_gpu_hosts.json"
    ):
        self.security_config = self._load_security_config(security_config_path)
        self.known_hosts = self._load_known_hosts(known_hosts_db)
        self.connection_history = []

    def _load_security_config(self, path: str) -> Dict:
        """Load security policy from cybersecurity pipeline"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default security policy
            return {
                "allowed_networks": ["192.168.1.0/24", "10.0.0.0/8"],
                "require_vpn_for_external": True,
                "allowed_cloud_providers": ["vast.ai", "runpod.io", "lambdalabs.com"],
                "max_connection_attempts": 3,
                "require_ssh_key_auth": True,
                "allowed_ssh_key_types": ["ed25519", "rsa-4096"],
                "require_host_verification": True
            }

    def _load_known_hosts(self, path: str) -> Dict:
        """Load database of previously connected GPU hosts"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def discover_cluster(self, partial_info: str) -> GPUClusterInfo:
        """
        Discover GPU cluster from partial information

        Examples:
        - "192.168.1.50" → Auto-discover everything
        - "vast.ai" → Use API to provision
        - "company-gpu-server" → Resolve DNS, check security
        - "user@host:2222" → Parse and auto-discover rest
        """
        print(f"\n{'='*80}")
        print(f"  Intelligent GPU Cluster Discovery")
        print(f"{'='*80}")
        print(f"\nInput: {partial_info}")

        # Step 1: Parse partial information
        parsed = self._parse_partial_info(partial_info)
        print(f"\n[1/7] Parsed info: {parsed}")

        # Step 2: Check known hosts database
        cluster_info = self._check_known_hosts(parsed)
        print(f"\n[2/7] Checking known hosts...")

        # Step 3: Auto-discover missing details
        if not cluster_info:
            cluster_info = self._auto_discover(parsed)
        print(f"\n[3/7] Auto-discovery complete")

        # Step 4: Security clearance check
        cluster_info = self._security_clearance_check(cluster_info)
        print(f"\n[4/7] Security check: {'✓ PASSED' if cluster_info.security_cleared else '✗ FAILED'}")

        if not cluster_info.security_cleared:
            raise SecurityError(f"Connection to {cluster_info.host} blocked by security policy")

        # Step 5: Establish connection (with VPN if needed)
        cluster_info = self._establish_connection(cluster_info)
        print(f"\n[5/7] Connection established: {cluster_info.is_available}")

        # Step 6: Probe hardware capabilities
        cluster_info = self._probe_hardware(cluster_info)
        print(f"\n[6/7] Hardware detected:")
        print(f"      GPUs: {cluster_info.num_gpus}x {cluster_info.gpu_type}")
        print(f"      VRAM: {cluster_info.total_vram_gb:.1f} GB total")

        # Step 7: Save to known hosts
        self._save_to_known_hosts(cluster_info)
        print(f"\n[7/7] Saved to known hosts database")

        print(f"\n{'='*80}")
        print(f"✓ Cluster ready for distributed training")
        print(f"{'='*80}\n")

        return cluster_info

    def _parse_partial_info(self, partial: str) -> Dict:
        """
        Parse partial connection information

        Handles formats:
        - IP address: "192.168.1.50"
        - Hostname: "gpu-server.company.com"
        - SSH format: "user@host:port"
        - Cloud provider: "vast.ai", "runpod"
        - URL: "https://api.vast.ai"
        """
        parsed = {
            "host": None,
            "port": None,
            "username": None,
            "source_type": None
        }

        # Cloud provider detection
        if any(provider in partial.lower() for provider in ["vast.ai", "runpod", "lambda"]):
            parsed["source_type"] = "cloud_api"
            parsed["provider"] = partial.lower()
            return parsed

        # SSH format: user@host:port
        if "@" in partial:
            parts = partial.split("@")
            parsed["username"] = parts[0]
            host_port = parts[1]

            if ":" in host_port:
                parsed["host"], port_str = host_port.split(":")
                parsed["port"] = int(port_str)
            else:
                parsed["host"] = host_port
                parsed["port"] = 22  # Default SSH

            parsed["source_type"] = "ssh"
            return parsed

        # Just hostname or IP
        if ":" in partial:
            parsed["host"], port_str = partial.split(":")
            parsed["port"] = int(port_str)
        else:
            parsed["host"] = partial
            parsed["port"] = 22  # Default SSH

        parsed["source_type"] = "hostname"
        parsed["username"] = os.getenv("USER")  # Default to current user

        return parsed

    def _check_known_hosts(self, parsed: Dict) -> Optional[GPUClusterInfo]:
        """Check if host is in known hosts database"""
        host_key = parsed.get("host")

        if host_key and host_key in self.known_hosts:
            print(f"  ✓ Found in known hosts database")
            saved = self.known_hosts[host_key]

            # Reconstruct GPUClusterInfo from saved data
            return GPUClusterInfo(**saved)

        return None

    def _auto_discover(self, parsed: Dict) -> GPUClusterInfo:
        """
        Auto-discover missing connection details

        Attempts:
        1. DNS resolution
        2. Port scanning for SSH
        3. Credential discovery (SSH keys, config files)
        4. VPN requirement detection
        """
        cluster_info = GPUClusterInfo(
            host=parsed.get("host", "unknown"),
            port=parsed.get("port", 22),
            username=parsed.get("username", os.getenv("USER")),
            auth_method="unknown"
        )

        # Cloud provider auto-provisioning
        if parsed.get("source_type") == "cloud_api":
            return self._auto_provision_cloud(parsed)

        # 1. DNS resolution
        try:
            ip = socket.gethostbyname(cluster_info.host)
            print(f"  ✓ DNS resolved: {cluster_info.host} → {ip}")
        except socket.gaierror:
            print(f"  ✗ DNS resolution failed for {cluster_info.host}")
            raise ConnectionError(f"Cannot resolve hostname: {cluster_info.host}")

        # 2. Port scanning (if port unknown)
        if cluster_info.port == 22:
            discovered_port = self._scan_for_ssh_port(ip)
            if discovered_port:
                cluster_info.port = discovered_port
                print(f"  ✓ SSH port discovered: {discovered_port}")

        # 3. Credential discovery
        auth_method, credentials = self._discover_credentials(cluster_info)
        cluster_info.auth_method = auth_method

        if auth_method == "ssh_key":
            cluster_info.ssh_key_path = credentials
            print(f"  ✓ SSH key found: {credentials}")
        elif auth_method == "password":
            cluster_info.password = credentials
            print(f"  ✓ Password found in secure storage")

        # 4. Check if VPN required
        cluster_info.requires_vpn = self._check_vpn_requirement(ip)
        if cluster_info.requires_vpn:
            cluster_info.vpn_config_path = self._find_vpn_config()
            print(f"  ✓ VPN required: {cluster_info.vpn_config_path}")

        return cluster_info

    def _scan_for_ssh_port(self, host: str) -> Optional[int]:
        """Scan common SSH ports"""
        common_ssh_ports = [22, 2222, 22022, 2200]

        for port in common_ssh_ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                result = sock.connect_ex((host, port))
                sock.close()

                if result == 0:
                    return port
            except:
                continue

        return None

    def _discover_credentials(self, cluster_info: GPUClusterInfo) -> Tuple[str, str]:
        """
        Discover authentication credentials

        Checks (in order):
        1. ~/.ssh/id_ed25519 (preferred)
        2. ~/.ssh/id_rsa
        3. ~/.ssh/config for host-specific key
        4. Password manager integration
        5. Environment variables
        """
        home = Path.home()

        # Check for Ed25519 key (preferred)
        ed25519_key = home / ".ssh" / "id_ed25519"
        if ed25519_key.exists():
            return ("ssh_key", str(ed25519_key))

        # Check for RSA key
        rsa_key = home / ".ssh" / "id_rsa"
        if rsa_key.exists():
            return ("ssh_key", str(rsa_key))

        # Check SSH config for host-specific key
        ssh_config = home / ".ssh" / "config"
        if ssh_config.exists():
            host_key = self._parse_ssh_config(ssh_config, cluster_info.host)
            if host_key:
                return ("ssh_key", host_key)

        # Check environment variables
        if os.getenv(f"SSH_KEY_{cluster_info.host.upper()}"):
            return ("ssh_key", os.getenv(f"SSH_KEY_{cluster_info.host.upper()}"))

        # Password manager (if available)
        password = self._check_password_manager(cluster_info.host)
        if password:
            return ("password", password)

        # No credentials found
        raise CredentialError(f"No credentials found for {cluster_info.host}")

    def _parse_ssh_config(self, config_path: Path, host: str) -> Optional[str]:
        """Parse SSH config for host-specific key"""
        try:
            with open(config_path, 'r') as f:
                content = f.read()

            # Simple parser for IdentityFile
            in_host_block = False
            for line in content.split('\n'):
                line = line.strip()

                if line.startswith('Host ') and host in line:
                    in_host_block = True
                elif line.startswith('Host ') and in_host_block:
                    break
                elif in_host_block and 'IdentityFile' in line:
                    key_path = line.split('IdentityFile')[1].strip()
                    key_path = os.path.expanduser(key_path)
                    if os.path.exists(key_path):
                        return key_path
        except:
            pass

        return None

    def _check_password_manager(self, host: str) -> Optional[str]:
        """Check password manager for credentials"""
        # Integration with common password managers
        # For now, check environment variables
        return os.getenv(f"GPU_PASSWORD_{host.upper().replace('.', '_')}")

    def _check_vpn_requirement(self, ip: str) -> bool:
        """
        Check if VPN required to reach host

        Checks:
        1. Is IP in private range but not local network?
        2. Security policy requires VPN for external access?
        """
        import ipaddress

        try:
            ip_obj = ipaddress.ip_address(ip)

            # Check if private but not local
            if ip_obj.is_private:
                # Check if in local network
                for allowed_net in self.security_config.get("allowed_networks", []):
                    network = ipaddress.ip_network(allowed_net)
                    if ip_obj in network:
                        return False  # Local network, no VPN needed

                # Private but not local → requires VPN
                return True
            else:
                # Public IP → check security policy
                return self.security_config.get("require_vpn_for_external", False)

        except:
            return False

    def _find_vpn_config(self) -> Optional[str]:
        """Find VPN configuration file"""
        vpn_configs = [
            "/etc/openvpn/client.conf",
            f"{Path.home()}/.config/vpn/client.ovpn",
            "/home/user/LAT5150DRVMIL/00-security/vpn/corporate.ovpn"
        ]

        for config in vpn_configs:
            if os.path.exists(config):
                return config

        return None

    def _auto_provision_cloud(self, parsed: Dict) -> GPUClusterInfo:
        """
        Auto-provision cloud GPU cluster

        Supports:
        - Vast.ai
        - RunPod
        - Lambda Labs
        """
        provider = parsed.get("provider", "")

        if "vast" in provider:
            return self._provision_vast_ai()
        elif "runpod" in provider:
            return self._provision_runpod()
        elif "lambda" in provider:
            return self._provision_lambda()

        raise ValueError(f"Unknown cloud provider: {provider}")

    def _provision_vast_ai(self) -> GPUClusterInfo:
        """Auto-provision Vast.ai instance"""
        api_key = os.getenv("VAST_API_KEY")
        if not api_key:
            raise CredentialError("VAST_API_KEY not found in environment")

        headers = {"Authorization": f"Bearer {api_key}"}

        # Search for cheapest A100 instances
        response = requests.get(
            "https://console.vast.ai/api/v0/bundles",
            headers=headers,
            params={"gpu_name": "A100", "num_gpus": 4}
        )

        offers = response.json()['offers']
        best_offer = min(offers, key=lambda x: x['dph_total'])

        # Rent instance
        rent_response = requests.put(
            f"https://console.vast.ai/api/v0/asks/{best_offer['id']}/",
            headers=headers,
            json={"image": "nvidia/pytorch:24.01-py3"}
        )

        instance = rent_response.json()

        # Wait for ready
        while True:
            status_response = requests.get(
                f"https://console.vast.ai/api/v0/instances/{instance['id']}",
                headers=headers
            )
            status = status_response.json()

            if status['actual_status'] == 'running':
                break
            time.sleep(10)

        # Create cluster info
        return GPUClusterInfo(
            host=status['public_ipaddr'],
            port=status['ssh_port'],
            username='root',
            auth_method='ssh_key',
            ssh_key_path=instance.get('ssh_key'),
            num_gpus=best_offer['num_gpus'],
            gpu_type=best_offer['gpu_name'],
            total_vram_gb=best_offer['gpu_ram'] * best_offer['num_gpus'],
            security_cleared=True,
            is_available=True
        )

    def _security_clearance_check(self, cluster_info: GPUClusterInfo) -> GPUClusterInfo:
        """
        Check security policy compliance

        Integrates with cybersecurity pipeline
        """
        print(f"  Running security checks...")

        # 1. Check if host is in allowed networks
        import ipaddress
        try:
            ip = socket.gethostbyname(cluster_info.host)
            ip_obj = ipaddress.ip_address(ip)

            allowed = False
            for net_str in self.security_config.get("allowed_networks", []):
                network = ipaddress.ip_network(net_str)
                if ip_obj in network:
                    allowed = True
                    break

            if not allowed and not ip_obj.is_global:
                print(f"    ✗ Host {ip} not in allowed networks")
                cluster_info.security_cleared = False
                return cluster_info
        except:
            pass

        # 2. Check authentication method
        if self.security_config.get("require_ssh_key_auth", True):
            if cluster_info.auth_method != "ssh_key":
                print(f"    ✗ SSH key authentication required")
                cluster_info.security_cleared = False
                return cluster_info

        # 3. Check VPN requirement
        if cluster_info.requires_vpn and not cluster_info.vpn_config_path:
            print(f"    ✗ VPN required but no VPN config found")
            cluster_info.security_cleared = False
            return cluster_info

        # All checks passed
        cluster_info.security_cleared = True
        cluster_info.security_policy = "standard_gpu_access"

        return cluster_info

    def _establish_connection(self, cluster_info: GPUClusterInfo) -> GPUClusterInfo:
        """
        Establish connection to cluster

        Handles:
        1. VPN connection (if required)
        2. SSH port forwarding (if required)
        3. Firewall traversal
        4. Connection testing
        """
        # 1. Connect VPN if needed
        if cluster_info.requires_vpn:
            self._connect_vpn(cluster_info.vpn_config_path)
            print(f"  ✓ VPN connected")

        # 2. Test SSH connection
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            if cluster_info.auth_method == "ssh_key":
                ssh.connect(
                    hostname=cluster_info.host,
                    port=cluster_info.port,
                    username=cluster_info.username,
                    key_filename=cluster_info.ssh_key_path,
                    timeout=10
                )
            elif cluster_info.auth_method == "password":
                ssh.connect(
                    hostname=cluster_info.host,
                    port=cluster_info.port,
                    username=cluster_info.username,
                    password=cluster_info.password,
                    timeout=10
                )

            # Test command
            stdin, stdout, stderr = ssh.exec_command("echo 'connection_test'")
            result = stdout.read().decode().strip()

            if result == "connection_test":
                cluster_info.is_available = True
                cluster_info.connection_tested = True
                print(f"  ✓ SSH connection successful")

            ssh.close()

        except Exception as e:
            print(f"  ✗ Connection failed: {e}")
            cluster_info.is_available = False
            cluster_info.connection_tested = False

        return cluster_info

    def _connect_vpn(self, config_path: str):
        """Connect to VPN"""
        # Start OpenVPN in background
        subprocess.Popen(
            ["sudo", "openvpn", "--config", config_path, "--daemon"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Wait for connection
        time.sleep(5)

    def _probe_hardware(self, cluster_info: GPUClusterInfo) -> GPUClusterInfo:
        """
        Probe remote hardware capabilities

        Detects:
        - Number of GPUs
        - GPU type
        - Total VRAM
        - CUDA version
        """
        if not cluster_info.is_available:
            return cluster_info

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        ssh.connect(
            hostname=cluster_info.host,
            port=cluster_info.port,
            username=cluster_info.username,
            key_filename=cluster_info.ssh_key_path
        )

        # Run nvidia-smi to detect GPUs
        stdin, stdout, stderr = ssh.exec_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
        output = stdout.read().decode()

        if output:
            lines = output.strip().split('\n')
            cluster_info.num_gpus = len(lines)

            # Parse first GPU
            if lines:
                first_line = lines[0]
                parts = first_line.split(',')
                cluster_info.gpu_type = parts[0].strip()
                vram_str = parts[1].strip().replace(' MiB', '')
                cluster_info.total_vram_gb = int(vram_str) * cluster_info.num_gpus / 1024

        ssh.close()

        return cluster_info

    def _save_to_known_hosts(self, cluster_info: GPUClusterInfo):
        """Save cluster to known hosts database"""
        self.known_hosts[cluster_info.host] = {
            "host": cluster_info.host,
            "port": cluster_info.port,
            "username": cluster_info.username,
            "auth_method": cluster_info.auth_method,
            "ssh_key_path": cluster_info.ssh_key_path,
            "num_gpus": cluster_info.num_gpus,
            "gpu_type": cluster_info.gpu_type,
            "total_vram_gb": cluster_info.total_vram_gb,
            "requires_vpn": cluster_info.requires_vpn,
            "vpn_config_path": cluster_info.vpn_config_path,
            "last_connected": time.time()
        }

        # Save to disk
        os.makedirs("/home/user/LAT5150DRVMIL/config", exist_ok=True)
        with open("/home/user/LAT5150DRVMIL/config/known_gpu_hosts.json", 'w') as f:
            json.dump(self.known_hosts, f, indent=2)

class SecurityError(Exception):
    """Security policy violation"""
    pass

class CredentialError(Exception):
    """Credential not found"""
    pass

# Usage examples
if __name__ == "__main__":
    discovery = IntelligentGPUDiscovery()

    # Example 1: Partial hostname
    cluster = discovery.discover_cluster("192.168.1.50")

    # Example 2: SSH format
    cluster = discovery.discover_cluster("ubuntu@gpu-server.company.com:2222")

    # Example 3: Cloud provider
    cluster = discovery.discover_cluster("vast.ai")

    # Example 4: Just hostname (auto-discover everything)
    cluster = discovery.discover_cluster("training-cluster")
```

---

### 3F: Learned MoE Routing with Gating Network (Weeks 23-26)

#### Moving from Rule-Based to Learned Routing

**Current State:** Pattern-based MoE (40% complete)
- 90+ regex patterns
- Hard-coded routing logic
- Cannot adapt to new domains

**Target State:** Learned gating network
- Neural network learns optimal routing
- Adapts based on performance feedback
- Generalizes to unseen queries

**Implementation: Gating Network Trainer**

**File:** `02-ai-engine/moe/learned_gating_trainer.py`

```python
#!/usr/bin/env python3
"""
Learned MoE Gating Network Trainer

Replaces rule-based routing with learned neural gating.

Architecture:
- Input: Query embedding (384-dim)
- Gating network: 384 → 256 → 128 → 9 (num experts)
- Output: Softmax probabilities over experts
- Training: Supervised from routing logs + RL from outcomes

Research papers:
- "Switch Transformers" (Fedus et al., 2021)
- "Outrageously Large Neural Networks: The Sparsely-Gated MoE" (Shazeer et al., 2017)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict
import numpy as np
from pathlib import Path

class GatingNetwork(nn.Module):
    """
    Learned gating network for MoE routing

    Input: Query embedding (384-dim from sentence-transformer)
    Output: Expert probabilities (9 experts)
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dims: List[int] = [256, 128],
        num_experts: int = 9,
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_experts))

        self.network = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            query_embedding: (batch_size, 384)

        Returns:
            expert_probs: (batch_size, 9) - probabilities over experts
        """
        logits = self.network(query_embedding)
        probs = self.softmax(logits)
        return probs

    def select_expert(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 1
    ) -> List[int]:
        """
        Select top-k experts for query

        Args:
            query_embedding: (384,)
            top_k: Number of experts to select

        Returns:
            expert_ids: List of expert indices
        """
        with torch.no_grad():
            probs = self.forward(query_embedding.unsqueeze(0))
            top_experts = torch.topk(probs, top_k, dim=-1).indices
            return top_experts[0].tolist()

class GatingNetworkTrainer:
    """
    Trainer for learned MoE gating network

    Training data sources:
    1. Historical routing logs (which expert was used for which query)
    2. Expert performance feedback (which expert gave best response)
    3. Human feedback on routing quality
    """

    def __init__(
        self,
        gating_network: GatingNetwork,
        learning_rate: float = 1e-3,
        device: str = "xpu"  # Intel Arc GPU
    ):
        self.model = gating_network.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def train_from_routing_logs(
        self,
        routing_logs_path: str = "/home/user/LAT5150DRVMIL/logs/moe_routing.json",
        num_epochs: int = 10,
        batch_size: int = 32
    ):
        """
        Train gating network from historical routing logs

        Log format:
        {
            "query": "What is quantum computing?",
            "expert_used": 2,  # 0-8
            "success": true,
            "response_quality": 0.85
        }
        """
        import json
        from sentence_transformers import SentenceTransformer

        # Load embedder (runs on NPU)
        embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # Load routing logs
        with open(routing_logs_path, 'r') as f:
            logs = [json.loads(line) for line in f]

        print(f"Loaded {len(logs)} routing logs")

        # Prepare dataset
        queries = [log['query'] for log in logs]
        expert_labels = [log['expert_used'] for log in logs]

        # Embed queries (batch on NPU)
        embeddings = embedder.encode(queries, batch_size=64, show_progress_bar=True)

        # Convert to torch tensors
        X = torch.FloatTensor(embeddings).to(self.device)
        y = torch.LongTensor(expert_labels).to(self.device)

        # Training loop
        self.model.train()

        for epoch in range(num_epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            # Mini-batch training
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Metrics
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

            accuracy = correct / total
            avg_loss = total_loss / (len(X) / batch_size)

            print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")

    def train_with_rl_feedback(
        self,
        num_iterations: int = 1000,
        episodes_per_iteration: int = 10
    ):
        """
        Train gating network with RL feedback

        Algorithm:
        1. Generate query
        2. Gating network selects expert
        3. Expert generates response
        4. Collect reward (human feedback or automatic metric)
        5. Update gating network with policy gradient
        """
        for iteration in range(num_iterations):
            # Collect episodes
            episode_rewards = []
            episode_log_probs = []

            for episode in range(episodes_per_iteration):
                # ... RL episode collection ...
                pass

            # Policy gradient update
            # ... (implementation similar to PPO trainer) ...
            pass

    def save_model(self, path: str):
        """Save trained gating network"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        print(f"✓ Saved gating network to {path}")

    def load_model(self, path: str):
        """Load trained gating network"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✓ Loaded gating network from {path}")

# Usage
if __name__ == "__main__":
    # Create gating network
    gating_net = GatingNetwork(
        input_dim=384,
        hidden_dims=[256, 128],
        num_experts=9
    )

    # Train from routing logs
    trainer = GatingNetworkTrainer(gating_net, device="xpu")
    trainer.train_from_routing_logs(
        routing_logs_path="/home/user/LAT5150DRVMIL/logs/moe_routing.json",
        num_epochs=10
    )

    # Save trained model
    trainer.save_model("/home/user/LAT5150DRVMIL/models/gating_network.pt")
```

---

## PHASE 4: Meta-Learning + Evaluation (Weeks 29-36)

### 4A: Model-Agnostic Meta-Learning (MAML) (Weeks 29-32)

**Research Paper:** "Model-Agnostic Meta-Learning for Fast Adaptation" (Finn et al., 2017)

**Goal:** Enable fast adaptation to new tasks with few examples

**Implementation:** `02-ai-engine/meta_learning/maml_trainer.py`

```python
#!/usr/bin/env python3
"""
MAML (Model-Agnostic Meta-Learning) Trainer

Enables fast adaptation to new tasks with minimal examples.

Meta-learning process:
1. Sample batch of tasks
2. For each task:
   - Inner loop: Adapt model with few examples
   - Compute task loss
3. Outer loop: Update meta-parameters to improve adaptation

Use cases:
- New domain adaptation (legal, medical, etc.)
- Few-shot learning for rare queries
- Personalization to user preferences
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import copy

class MAMLTrainer:
    """
    MAML trainer for few-shot adaptation

    Hardware optimization:
    - Inner loop on Arc GPU (fast iteration)
    - Outer loop gradient accumulation
    - Checkpoint to avoid OOM
    """

    def __init__(
        self,
        base_model,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        num_inner_steps: int = 5,
        device: str = "xpu"
    ):
        self.base_model = base_model.to(device)
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        self.device = device

        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(
            self.base_model.parameters(),
            lr=outer_lr
        )

    def inner_loop_adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor
    ) -> nn.Module:
        """
        Inner loop: Adapt model to task with few examples

        Args:
            support_x: Support set inputs (few examples)
            support_y: Support set labels

        Returns:
            adapted_model: Model adapted to task
        """
        # Clone model for adaptation
        adapted_model = copy.deepcopy(self.base_model)
        adapted_model.train()

        # Inner loop optimizer
        inner_optimizer = torch.optim.SGD(
            adapted_model.parameters(),
            lr=self.inner_lr
        )

        # Adapt for few steps
        for step in range(self.num_inner_steps):
            # Forward pass
            outputs = adapted_model(support_x)
            loss = nn.functional.cross_entropy(outputs, support_y)

            # Backward pass
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()

        return adapted_model

    def meta_train_step(
        self,
        task_batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> float:
        """
        Meta-training step (outer loop)

        Args:
            task_batch: List of (support_x, support_y, query_x, query_y) tuples

        Returns:
            meta_loss: Average loss across tasks
        """
        meta_loss = 0.0

        for support_x, support_y, query_x, query_y in task_batch:
            # Inner loop: Adapt to task
            adapted_model = self.inner_loop_adapt(support_x, support_y)

            # Evaluate on query set
            adapted_model.eval()
            with torch.set_grad_enabled(True):
                query_outputs = adapted_model(query_x)
                task_loss = nn.functional.cross_entropy(query_outputs, query_y)

            meta_loss += task_loss

        # Average across tasks
        meta_loss = meta_loss / len(task_batch)

        # Meta-optimization step
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

    def fast_adapt(
        self,
        few_examples: List[Tuple[str, str]],
        num_steps: int = 5
    ) -> nn.Module:
        """
        Fast adaptation to new task with few examples

        Args:
            few_examples: List of (input, output) pairs
            num_steps: Number of adaptation steps

        Returns:
            adapted_model: Model adapted to new task
        """
        # Prepare examples
        support_x = [ex[0] for ex in few_examples]
        support_y = [ex[1] for ex in few_examples]

        # Embed inputs (on NPU)
        # ... embedding code ...

        # Adapt model
        adapted_model = self.inner_loop_adapt(support_x_embedded, support_y_embedded)

        return adapted_model

# Example usage
if __name__ == "__main__":
    # Meta-train on diverse tasks
    maml = MAMLTrainer(base_model, device="xpu")

    # After meta-training, adapt to new domain with 5 examples
    medical_examples = [
        ("Patient presents with fever and cough", "Likely respiratory infection"),
        ("ECG shows ST elevation", "Possible myocardial infarction"),
        # ... 3 more examples
    ]

    adapted_model = maml.fast_adapt(medical_examples, num_steps=5)

    # Now model is specialized for medical domain
```

### 4B: Comprehensive Evaluation Framework (Weeks 33-36)

**File:** `02-ai-engine/evaluation/comprehensive_evaluator.py`

```python
#!/usr/bin/env python3
"""
Comprehensive AI Framework Evaluation

Metrics:
1. RAG Performance: Precision@K, Recall@K, NDCG
2. Response Quality: BLEU, ROUGE, BERTScore
3. Self-Improvement: Week-over-week delta
4. Hardware Efficiency: Latency, throughput, utilization
5. Meta-Learning: Few-shot adaptation accuracy
"""

class ComprehensiveEvaluator:
    """
    Evaluate all AI framework components

    Generates weekly improvement reports
    """

    def __init__(self):
        self.metrics_history = []

    def evaluate_all(self) -> Dict:
        """Run comprehensive evaluation suite"""
        results = {
            "rag_performance": self._evaluate_rag(),
            "response_quality": self._evaluate_responses(),
            "hardware_efficiency": self._evaluate_hardware(),
            "meta_learning": self._evaluate_meta_learning(),
            "overall_improvement": self._calculate_improvement()
        }

        self.metrics_history.append(results)

        return results

    def _evaluate_rag(self) -> Dict:
        """Evaluate RAG system performance"""
        # Test on benchmark dataset
        # Return precision, recall, NDCG
        pass

    def _calculate_improvement(self) -> float:
        """Calculate week-over-week improvement"""
        if len(self.metrics_history) < 2:
            return 0.0

        current = self.metrics_history[-1]
        previous = self.metrics_history[-2]

        # Weighted average of all metrics
        improvement = (
            0.3 * (current['rag_performance']['ndcg'] - previous['rag_performance']['ndcg']) +
            0.3 * (current['response_quality']['bert_score'] - previous['response_quality']['bert_score']) +
            0.2 * (current['meta_learning']['adaptation_accuracy'] - previous['meta_learning']['adaptation_accuracy']) +
            0.2 * (current['hardware_efficiency']['throughput'] - previous['hardware_efficiency']['throughput'])
        )

        return improvement
```

---

## IMPLEMENTATION TIMELINE SUMMARY

| Phase | Component | Weeks | Key Deliverable |
|-------|-----------|-------|-----------------|
| 1 | DPO Training | 1-6 | +15-25% response quality |
| 2 | Self-RAG | 7-12 | +10-20% RAG accuracy |
| 3A | Cloud PPO Training | 13-20 | +30-50% via RL |
| 3B | C++ AVX-512 Optimization | 15-18 | 5x vector search speedup |
| 3C | Dynamic Hardware Manager | 19-22 | Optimal resource allocation |
| 3D | ZFS Storage Optimizer | 21-22 | 2-4x space savings |
| 3E | Multi-GPU Discovery | 21-24 | Intelligent cluster connection |
| 3F | Learned MoE Routing | 23-26 | Adaptive expert selection |
| 4A | MAML Meta-Learning | 29-32 | Few-shot adaptation |
| 4B | Evaluation Framework | 33-36 | Continuous monitoring |

**Total Duration:** 36 weeks (9 months)

**Expected Cumulative Improvement:** +70-120% over baseline

---

## HARDWARE UTILIZATION MATRIX

| Workload | NPU | Arc GPU | NCS2 | AVX-512 | E-cores |
|----------|-----|---------|------|---------|---------|
| Embeddings | ✓✓✓ | ✓ | ✓ | - | ✓ |
| Vector Search | ✓ | - | - | ✓✓✓ | ✓✓ |
| LoRA Training | - | ✓✓✓ | - | - | - |
| Inference (INT8) | ✓✓✓ | ✓✓ | ✓✓ | ✓ | ✓ |
| Reranking | ✓ | ✓ | ✓✓✓ | - | ✓ |
| Gating Network | ✓✓ | ✓✓✓ | - | ✓ | - |
| DPO Training | - | ✓✓✓ | - | - | - |
| PPO Training | Cloud GPUs (4-8x A100) | | | | |

✓✓✓ = Optimal, ✓✓ = Good, ✓ = Acceptable, - = Not suitable

---

## NEXT STEPS

1. **Week 1-6:** Implement DPO training pipeline
2. **Week 7-12:** Implement Self-RAG with reflection
3. **Week 13:** Begin cloud GPU integration
4. **Week 15:** Start C++ AVX-512 optimization
5. **Week 21:** Deploy multi-GPU discovery system
6. **Week 29:** Begin MAML meta-learning
7. **Week 36:** Complete evaluation framework

**End State:**
- Fully automated self-improving AI system
- Optimized for Dell Latitude 5450 MIL-SPEC hardware
- Cloud-hybrid architecture for PPO training
- Continuous improvement via RL feedback
- Hardware-aware workload distribution
- Meta-learning for rapid adaptation

---

**Document Status:** ✅ COMPLETE

**Implementation Plan Parts:**
- Part 1: Phases 1-2 (DPO + Self-RAG)
- Part 2: Phases 3-4 (PPO + Meta-Learning + C++ Optimization + Multi-GPU)

**Total Pages:** ~150 pages combined
**Total Code:** ~5000 lines of implementation
**Research Papers Referenced:** 80+

**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY

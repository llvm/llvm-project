# HERETIC Integration Plan - LAT5150DRVMIL AI Engine

**Date:** 2025-11-16
**Repository:** https://github.com/p-e-w/heretic
**Target System:** 02-ai-engine (Enhanced AI Engine + DSMIL AI Engine)

---

## PHASE 1: FULL ENUMERATION

### 1.1 Core Methods from Heretic

#### A. Model Manipulation Methods (model.py)

**Class: `Model`**

1. **`__init__(settings: Settings)`** - Initialize model with dtype fallback
2. **`reload_model()`** - Fresh model reload with GPU cache clearing
3. **`get_layers() -> ModuleList`** - Extract transformer layers (multimodal fallback)
4. **`get_layer_matrices(layer_index: int) -> dict[str, list[Tensor]]`** - Extract weight matrices
   - Targets: `attn.o_proj`, `mlp.down_proj`
   - MoE support: Qwen3, Phi-3.5, gpt-oss variants
5. **`get_abliterable_components() -> list[str]`** - Return ["attn", "mlp"]
6. **`abliterate(refusal_directions, direction_index, parameters)`** - **CORE ALGORITHM**
   - Direction interpolation (fractional indexing)
   - Layer-wise weight calculation with distance-based kernel
   - Orthogonal projection: `W_new = W_old - α * (r ⊗ r) @ W_old`

**Inference Methods:**
7. **`generate(prompts, **kwargs)`** - Tokenize and generate
8. **`get_responses(prompts) -> list[str]`** - Single-batch response generation
9. **`get_responses_batched(prompts) -> list[str]`** - Batched inference
10. **`get_chat(prompt) -> list[dict]`** - Format chat messages
11. **`stream_chat_response(chat) -> str`** - Stream response token-by-token

**Analysis Methods:**
12. **`get_residuals(prompts) -> Tensor`** - Extract hidden states [n_prompts, n_layers, hidden_size]
13. **`get_residuals_batched(prompts) -> Tensor`** - Batched residual extraction
14. **`get_logprobs(prompts) -> Tensor`** - Compute log-probability distributions
15. **`get_logprobs_batched(prompts) -> Tensor`** - Batched logprob extraction

**Dataclass:**
16. **`AbliterationParameters`** - max_weight, max_weight_position, min_weight, min_weight_distance

#### B. Evaluation Methods (evaluator.py)

**Class: `Evaluator`**

17. **`__init__(settings, model)`** - Load evaluation prompts, compute baselines
18. **`is_refusal(response: str) -> bool`** - String-based refusal detection
19. **`count_refusals() -> int`** - Count refusals on harmful prompts
20. **`get_score() -> tuple`** - Return (normalized_score, kl_divergence, refusal_count)
    - Uses `F.kl_div()` for KL divergence measurement

#### C. Optimization Methods (main.py)

21. **`main()`** - Entry point with Rich traceback
22. **`run()`** - Main orchestration:
    - Accelerator detection
    - Prompt loading
    - Batch size optimization (exponential search)
    - Refusal direction calculation
    - Optuna study creation
    - Trial selection and actions
23. **`objective(trial) -> tuple[float, float]`** - Multi-objective optimization:
    - Sample parameters: direction_scope, direction_index, component parameters
    - Apply abliteration
    - Evaluate and return score

#### D. Utility Methods (utils.py)

24. **`format_duration(seconds) -> str`** - Human-readable duration
25. **`load_prompts(spec) -> list[str]`** - Load datasets from HuggingFace
26. **`batchify(items, batch_size) -> list[list]`** - Divide into chunks
27. **`empty_cache()`** - Clear GPU memory (CUDA, XPU, MLU, SDAA, MUSA, NPU)
28. **`get_trial_parameters(trial) -> dict`** - Extract optimization metadata
29. **`get_readme_intro(base_model, trial, evaluator) -> str`** - Generate model card

#### E. Configuration Methods (config.py)

30. **`DatasetSpecification`** - dataset, split, column
31. **`Settings`** - Complete configuration with TOML/env/CLI support
    - Model loading, generation, optimization, refusal detection

### 1.2 Core Algorithms

#### Algorithm 1: Refusal Direction Calculation
```python
# Difference-of-means per layer
good_residuals = model.get_residuals_batched(good_prompts)  # [n_good, n_layers, hidden_size]
bad_residuals = model.get_residuals_batched(bad_prompts)    # [n_bad, n_layers, hidden_size]

refusal_directions = bad_residuals.mean(dim=0) - good_residuals.mean(dim=0)  # [n_layers, hidden_size]
refusal_directions = F.normalize(refusal_directions, p=2, dim=-1)  # L2 normalize per layer
```

**Formula:** `r_l = normalize(mean(h_bad,l) - mean(h_good,l))`

#### Algorithm 2: Fractional Direction Interpolation
```python
if direction_index is not None:
    floor_idx = int(direction_index)
    ceil_idx = floor_idx + 1
    weight = direction_index - floor_idx

    refusal_direction = (1 - weight) * refusal_directions[floor_idx] + weight * refusal_directions[ceil_idx]
else:
    refusal_direction = refusal_directions.mean(dim=0)  # Global direction
```

#### Algorithm 3: Distance-Based Weight Kernel
```python
for layer_idx in range(n_layers):
    distance = abs(layer_idx - params.max_weight_position)

    if distance > params.min_weight_distance:
        continue  # Skip distant layers

    # Linear interpolation from max_weight to min_weight
    layer_weight = params.max_weight + (params.min_weight - params.max_weight) * (distance / params.min_weight_distance)
```

#### Algorithm 4: Orthogonal Projection (Core Abliteration)
```python
# Normalize refusal direction
layer_refusal_direction = F.normalize(refusal_direction, p=2, dim=-1)

# Create projection matrix (outer product)
projector = torch.outer(layer_refusal_direction, layer_refusal_direction)  # [hidden_size, hidden_size]

# Apply to all matrices in layer
for matrix in layer_matrices:
    matrix.sub_(layer_weight * (projector @ matrix))  # In-place subtraction
```

**Formula:** `W_new = W_old - α * (r ⊗ r) @ W_old`

#### Algorithm 5: Multi-Objective Optimization
```python
study = optuna.create_study(
    directions=["minimize", "minimize"],  # [KL divergence, refusal rate]
    sampler=TPESampler(
        n_startup_trials=60,      # Random exploration phase
        multivariate=True         # Model parameter correlations
    )
)

study.optimize(objective, n_trials=200)
pareto_trials = study.best_trials  # Non-dominated solutions
```

#### Algorithm 6: Batch Size Optimization
```python
batch_size = 1
best_throughput = 0

while batch_size <= max_batch_size:
    try:
        tokens_per_sec = benchmark(batch_size)
        if tokens_per_sec > best_throughput:
            best_throughput = tokens_per_sec
            optimal_batch_size = batch_size
        batch_size *= 2  # Exponential growth
    except OutOfMemoryError:
        break
```

### 1.3 Key Features to Integrate

1. **Directional Ablation** - Remove safety constraints via orthogonal projection
2. **Multi-Objective Optimization** - Balance refusal removal vs model fidelity
3. **Flexible Weight Kernels** - Distance-based ablation scheduling
4. **Fractional Indexing** - Interpolate between layer directions
5. **Component-Specific Tuning** - Separate parameters for attention/MLP
6. **MoE Support** - Handle Mixture of Experts architectures
7. **Multi-Accelerator Support** - CUDA, XPU, MLU, NPU compatibility
8. **Refusal Detection** - String-based and LLM-based methods
9. **KL Divergence Metrics** - Measure model fidelity degradation
10. **Automatic Batch Optimization** - Find optimal batch size dynamically

---

## PHASE 2: INTEGRATION ARCHITECTURE

### 2.1 New Modules to Create

#### Module 1: `heretic_abliteration.py`
**Purpose:** Core abliteration engine
**Exports:**
- `AbliterationParameters` (dataclass)
- `HereticModelWrapper` (class) - Wraps Ollama/local models with abliteration
- `RefusalDirectionCalculator` - Calculate refusal directions from datasets
- `ModelAbliterator` - Apply abliteration with flexible kernels

#### Module 2: `heretic_optimizer.py`
**Purpose:** Optuna-based parameter optimization
**Exports:**
- `HereticOptimizer` (class) - Multi-objective optimization
- `OptimizationObjective` - Callable for Optuna trials
- `TrialSelector` - Interactive trial selection
- `ParameterSampler` - Sample abliteration parameters

#### Module 3: `heretic_evaluator.py`
**Purpose:** Model evaluation and metrics
**Exports:**
- `RefusalDetector` - String-based + LLM-based refusal detection
- `ModelEvaluator` - KL divergence, refusal counting
- `BenchmarkSuite` - Comprehensive model testing

#### Module 4: `heretic_datasets.py`
**Purpose:** Dataset management for abliteration
**Exports:**
- `DatasetSpecification` - Config for datasets
- `PromptLoader` - Load from HuggingFace/local
- `DatasetRegistry` - Manage harmless/harmful datasets

#### Module 5: `heretic_config.py`
**Purpose:** Configuration management
**Exports:**
- `HereticSettings` - Pydantic settings model
- `ConfigLoader` - Load from TOML/env/CLI

### 2.2 Integration Points

#### A. Enhanced AI Engine Integration

**File:** `enhanced_ai_engine.py`

**New Capabilities:**
1. **Model Abliteration** - Apply abliteration before inference
2. **Refusal Analysis** - Detect and analyze refusals
3. **Safety Constraint Removal** - Optimize models for uncensored output
4. **Model Fidelity Tracking** - Monitor KL divergence over time

**New Methods:**
```python
class EnhancedAIEngine:
    def abliterate_model(self, model_name: str, optimization_trials: int = 200) -> dict
    def get_refusal_directions(self, model_name: str) -> torch.Tensor
    def evaluate_model_safety(self, model_name: str) -> dict
    def apply_custom_abliteration(self, model_name: str, parameters: dict) -> bool
```

#### B. DSMIL AI Engine Integration

**File:** `dsmil_ai_engine.py`

**New Capabilities:**
1. **Hardware-Attested Abliteration** - TPM attestation of abliterated models
2. **Quantum-Encrypted Refusal Directions** - Secure storage of refusal vectors
3. **DSMIL-Accelerated Optimization** - Use NPU/GPU for faster optimization
4. **Military-Grade Model Hardening** - Apply abliteration with integrity checks

**New Methods:**
```python
class DSMILAIEngine:
    def hardware_attested_abliterate(self, model: str, parameters: dict) -> dict
    def secure_store_refusal_directions(self, model: str, directions: Tensor) -> str
    def dsmil_accelerated_optimize(self, model: str, trials: int) -> dict
```

#### C. MCP Server Integration

**File:** `heretic_mcp_server.py` (NEW)

**MCP Endpoints:**
1. `abliterate_model` - Run full abliteration workflow
2. `optimize_abliteration` - Run Optuna optimization
3. `evaluate_model` - Get refusal counts and KL divergence
4. `list_abliterated_models` - List saved abliterated models
5. `get_refusal_directions` - Retrieve refusal direction vectors
6. `apply_abliteration` - Apply pre-computed parameters

#### D. CLI Integration

**File:** `heretic_cli.py` (NEW)

**Commands:**
```bash
# Run full abliteration
python heretic_cli.py abliterate --model uncensored_code --trials 200

# Evaluate current model
python heretic_cli.py evaluate --model uncensored_code

# Apply saved abliteration
python heretic_cli.py apply --model uncensored_code --params saved_params.json

# Interactive optimization
python heretic_cli.py optimize --model uncensored_code --interactive
```

### 2.3 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Request (CLI/API/MCP)                    │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
        ┌───────▼────────┐            ┌────────▼────────┐
        │ Enhanced AI    │            │ DSMIL AI        │
        │ Engine         │            │ Engine          │
        │                │            │                 │
        │ +abliterate()  │            │ +hw_attest()    │
        └───────┬────────┘            └────────┬────────┘
                │                              │
                └──────────┬───────────────────┘
                           │
        ┌──────────────────▼──────────────────┐
        │      HereticModelWrapper             │
        │  - Load model (Ollama/HF)            │
        │  - Extract layers & matrices         │
        │  - Apply abliteration                │
        └──────────────────┬──────────────────┘
                           │
        ┌──────────────────▼──────────────────┐
        │    RefusalDirectionCalculator        │
        │  - Load harmless/harmful prompts     │
        │  - Extract residual activations      │
        │  - Compute difference-of-means       │
        │  - Normalize (L2 per layer)          │
        └──────────────────┬──────────────────┘
                           │
        ┌──────────────────▼──────────────────┐
        │       HereticOptimizer               │
        │  - Create Optuna study (TPE)         │
        │  - Sample parameters (multivariate)  │
        │  - Evaluate each trial               │
        │  - Return Pareto-optimal solutions   │
        └──────────────────┬──────────────────┘
                           │
        ┌──────────────────▼──────────────────┐
        │       ModelAbliterator               │
        │  - Fractional direction interp       │
        │  - Distance-based weight kernel      │
        │  - Orthogonal projection             │
        │  - In-place weight modification      │
        └──────────────────┬──────────────────┘
                           │
        ┌──────────────────▼──────────────────┐
        │       ModelEvaluator                 │
        │  - Count refusals (harmful prompts)  │
        │  - Compute KL divergence (harmless)  │
        │  - Return (score, kl_div, refusals)  │
        └──────────────────┬──────────────────┘
                           │
                ┌──────────▼──────────┐
                │ Save/Deploy Model    │
                │ - Local disk         │
                │ - Ollama registry    │
                │ - HuggingFace Hub    │
                └──────────────────────┘
```

### 2.4 Configuration Schema

**File:** `/home/user/LAT5150DRVMIL/02-ai-engine/heretic_config.toml`

```toml
[heretic]
enabled = true
default_trials = 200
startup_trials = 60
max_batch_size = 128

[heretic.datasets]
good_prompts_dataset = "mlabonne/harmless_alpaca"
good_prompts_split = "train[:400]"
good_prompts_column = "text"

bad_prompts_dataset = "mlabonne/harmful_behaviors"
bad_prompts_split = "train[:400]"
bad_prompts_column = "text"

good_eval_dataset = "mlabonne/harmless_alpaca"
good_eval_split = "test[:100]"
good_eval_column = "text"

bad_eval_dataset = "mlabonne/harmful_behaviors"
bad_eval_split = "test[:100]"
bad_eval_column = "text"

[heretic.refusal_markers]
markers = [
    "sorry", "i can't", "i cannot", "i won't", "i will not",
    "i'm unable", "i am unable", "i'm an ai", "i am an ai",
    "as an ai", "ai assistant", "i'm designed to", "i am designed to",
    "i'm programmed to", "i am programmed to", "violat", "prohibit",
    "illegal", "harmful", "inappropriate", "unethical", "ethical boundaries"
]

[heretic.optimization]
kl_divergence_scale = 1.0
multivariate = true

[heretic.models]
# Models to abliterate
uncensored_code = { enabled = true, priority = "high" }
large = { enabled = true, priority = "medium" }
quality_code = { enabled = false, priority = "low" }

[heretic.storage]
abliterated_models_dir = "/home/user/LAT5150DRVMIL/02-ai-engine/abliterated_models"
refusal_directions_dir = "/home/user/LAT5150DRVMIL/02-ai-engine/refusal_directions"
optimization_results_dir = "/home/user/LAT5150DRVMIL/02-ai-engine/optimization_results"
```

---

## PHASE 3: DEEP INTEGRATION STRATEGY

### 3.1 Conversation Manager Integration

**Capability:** Track abliteration history across conversations

```python
# In conversation_manager.py
class Conversation:
    abliteration_history: List[dict] = []  # Track all abliterations
    current_abliteration_params: Optional[dict] = None
    refusal_directions: Optional[str] = None  # Path to saved directions
```

### 3.2 RAG System Integration

**Capability:** Index abliteration research papers and techniques

```python
# Auto-index heretic documentation and papers
rag_system.add_document("HERETIC_TECHNICAL_REPORT.md")
rag_system.add_document("abliteration_research_papers/")

# Query during optimization
relevant_papers = rag_system.query("optimal abliteration parameters for 34B models")
```

### 3.3 Hierarchical Memory Integration

**Capability:** Store refusal directions in long-term memory

```python
# Store refusal directions as memory blocks
hierarchical_memory.add_block(
    content=f"Refusal directions for {model_name}",
    block_type="refusal_direction",
    importance=1.0,
    metadata={
        "model": model_name,
        "directions_shape": str(refusal_directions.shape),
        "path": save_path
    }
)
```

### 3.4 DSMIL Hardware Integration

**Capability:** TPM attestation of abliterated models

```python
# Generate attestation for abliterated model
attestation = dsmil_integrator.attest_abliteration(
    original_model=model_name,
    abliteration_params=parameters,
    refusal_directions_hash=hashlib.sha256(refusal_directions.tobytes()).hexdigest()
)

# Verify model hasn't been tampered with
verified = dsmil_integrator.verify_abliterated_model(
    model_path=abliterated_model_path,
    expected_attestation=attestation
)
```

### 3.5 Quantum Crypto Integration

**Capability:** Encrypt refusal directions with post-quantum crypto

```python
# Encrypt refusal directions before storage
encrypted_directions = quantum_crypto.encrypt(
    data=refusal_directions.tobytes(),
    key_purpose="refusal_directions_storage"
)

# Decrypt when needed
decrypted_directions = quantum_crypto.decrypt(
    encrypted_data=encrypted_directions,
    key_purpose="refusal_directions_storage"
)
```

### 3.6 Self-Improvement Integration

**Capability:** Learn optimal abliteration strategies over time

```python
# Learn from successful abliterations
self_improvement.learn_from_interaction(
    insight_type="abliteration_success",
    content=f"Model {model_name} abliterated with KL divergence {kl_div:.3f}",
    confidence=0.9,
    actionable=True,
    metadata={
        "parameters": parameters,
        "refusal_count": refusal_count,
        "kl_divergence": kl_div
    }
)

# Suggest improvements
suggestions = self_improvement.suggest_improvements("abliteration")
```

### 3.7 MCP Server Ecosystem Integration

**New MCP Server:** `heretic-abliteration-mcp`

**Tools:**
1. `abliterate_model` - Full abliteration workflow
2. `optimize_parameters` - Run Optuna optimization
3. `evaluate_safety` - Check refusal rate and KL divergence
4. `list_abliterated` - List all abliterated models
5. `compare_models` - Compare original vs abliterated
6. `export_to_hf` - Upload to HuggingFace Hub

**Resources:**
1. `abliterated_models://` - Browse abliterated models
2. `refusal_directions://` - Access refusal direction vectors
3. `optimization_results://` - View optimization trials

---

## PHASE 4: IMPLEMENTATION ROADMAP

### 4.1 Sprint 1: Core Abliteration Engine (Day 1-2)

**Tasks:**
1. ✅ Create `heretic_abliteration.py` with:
   - `AbliterationParameters` dataclass
   - `HereticModelWrapper` class
   - `RefusalDirectionCalculator` class
   - `ModelAbliterator` class

2. ✅ Create `heretic_config.py` with:
   - `HereticSettings` Pydantic model
   - `ConfigLoader` for TOML/env/CLI

3. ✅ Create `heretic_datasets.py` with:
   - `DatasetSpecification` class
   - `PromptLoader` class
   - `DatasetRegistry` class

4. ✅ Unit tests for core algorithms

### 4.2 Sprint 2: Optimization & Evaluation (Day 3-4)

**Tasks:**
1. ✅ Create `heretic_optimizer.py` with:
   - `HereticOptimizer` class
   - `OptimizationObjective` callable
   - `TrialSelector` for interactive selection
   - `ParameterSampler` for Optuna

2. ✅ Create `heretic_evaluator.py` with:
   - `RefusalDetector` (string-based + LLM-based)
   - `ModelEvaluator` (KL divergence, refusal counting)
   - `BenchmarkSuite` for comprehensive testing

3. ✅ Integration tests for optimization loop

### 4.3 Sprint 3: AI Engine Integration (Day 5-6)

**Tasks:**
1. ✅ Modify `enhanced_ai_engine.py`:
   - Add `abliterate_model()` method
   - Add `get_refusal_directions()` method
   - Add `evaluate_model_safety()` method
   - Add `apply_custom_abliteration()` method

2. ✅ Modify `dsmil_ai_engine.py`:
   - Add `hardware_attested_abliterate()` method
   - Add `secure_store_refusal_directions()` method
   - Add `dsmil_accelerated_optimize()` method

3. ✅ Update `models.json`:
   - Add abliteration configuration section
   - Add refusal direction paths
   - Add optimization parameters

### 4.4 Sprint 4: MCP & CLI Interfaces (Day 7-8)

**Tasks:**
1. ✅ Create `heretic_mcp_server.py`:
   - Implement all 6 MCP endpoints
   - Add resource providers
   - Add streaming support for optimization

2. ✅ Create `heretic_cli.py`:
   - `abliterate` command
   - `optimize` command
   - `evaluate` command
   - `apply` command
   - Interactive mode

3. ✅ Integration with existing CLI tools

### 4.5 Sprint 5: Deep System Integration (Day 9-10)

**Tasks:**
1. ✅ Conversation Manager integration
2. ✅ RAG System integration
3. ✅ Hierarchical Memory integration
4. ✅ DSMIL Hardware integration
5. ✅ Quantum Crypto integration
6. ✅ Self-Improvement integration

### 4.6 Sprint 6: Testing & Documentation (Day 11-12)

**Tasks:**
1. ✅ Comprehensive test suite:
   - Unit tests for all modules
   - Integration tests for workflows
   - Performance benchmarks
   - Security tests

2. ✅ Documentation:
   - API reference
   - User guide
   - Tutorial notebooks
   - Architecture diagrams

3. ✅ Example workflows:
   - Basic abliteration
   - Custom optimization
   - Hardware-attested abliteration
   - Encrypted refusal storage

---

## PHASE 5: SUCCESS METRICS

### 5.1 Functional Metrics

- ✅ Successfully abliterate all 5 models in `models.json`
- ✅ Achieve <5% refusal rate on harmful prompts
- ✅ Maintain KL divergence <0.5 on harmless prompts
- ✅ Complete optimization in <60 minutes for 8B models
- ✅ All MCP endpoints functional
- ✅ CLI commands working end-to-end

### 5.2 Integration Metrics

- ✅ Abliteration history tracked in conversations
- ✅ Refusal directions indexed in RAG
- ✅ TPM attestation generated for abliterated models
- ✅ Refusal directions encrypted with quantum crypto
- ✅ Self-improvement learns from abliteration patterns
- ✅ MCP ecosystem fully integrated

### 5.3 Performance Metrics

- ✅ Batch size auto-optimization working
- ✅ Multi-accelerator support (CUDA, NPU)
- ✅ Memory-efficient refusal direction storage
- ✅ Fast KL divergence computation
- ✅ Parallel trial evaluation (if possible)

---

## PHASE 6: RISK MITIGATION

### 6.1 Technical Risks

**Risk:** Model corruption during abliteration
**Mitigation:** Always reload fresh model before each trial

**Risk:** Out-of-memory errors on large models
**Mitigation:** Automatic batch size optimization, gradient checkpointing

**Risk:** Incompatible model architectures
**Mitigation:** Architecture detection, fallback strategies

### 6.2 Security Risks

**Risk:** Abliterated models used for harmful purposes
**Mitigation:** TPM attestation, audit logging, user authorization

**Risk:** Refusal directions leaked
**Mitigation:** Quantum encryption, secure storage, access controls

### 6.3 Operational Risks

**Risk:** Long optimization times
**Mitigation:** Distributed optimization, surrogate models, early stopping

**Risk:** Poor abliteration quality
**Mitigation:** Multi-objective optimization, Pareto frontier selection, manual review

---

## APPENDIX A: Mathematical Foundations

### Refusal Direction Calculation

Given:
- Good prompts: `{g_1, g_2, ..., g_N}`
- Bad prompts: `{b_1, b_2, ..., b_M}`

For each layer `l`:
1. Extract residuals: `h_good,l,i = model.layers[l](g_i)[-1]`
2. Extract residuals: `h_bad,l,j = model.layers[l](b_j)[-1]`
3. Compute means: `μ_good,l = (1/N) Σ h_good,l,i`
4. Compute means: `μ_bad,l = (1/M) Σ h_bad,l,j`
5. Refusal direction: `r_l = (μ_bad,l - μ_good,l) / ||μ_bad,l - μ_good,l||_2`

### Orthogonal Projection

For weight matrix `W` at layer `l`:
1. Refusal direction: `r` (unit vector)
2. Projection matrix: `P = r ⊗ r = r × r^T`
3. Abliteration weight: `α_l` (layer-specific)
4. Updated weights: `W' = W - α_l × P × W = W - α_l × (r ⊗ r) × W`

Effect: Removes component of `W` aligned with `r`, preserves orthogonal components

### KL Divergence

For distributions `P` (original) and `Q` (abliterated):
```
KL(P || Q) = Σ P(x) log(P(x) / Q(x))
```

In code (log-space):
```python
kl_div = F.kl_div(
    input=log_Q,        # log-probabilities from abliterated model
    target=log_P,       # log-probabilities from original model
    log_target=True,    # Both inputs are in log-space
    reduction="batchmean"
)
```

---

## APPENDIX B: Dependencies

**New Dependencies:**
```toml
[project.dependencies]
# Core heretic functionality
optuna = ">=4.5.0"              # Bayesian optimization
pydantic-settings = ">=2.10.1"  # Configuration management
rich = ">=14.1.0"               # Pretty console output
questionary = ">=2.1.1"         # Interactive prompts

# Already have from existing system
transformers = ">=4.55.2"
torch = ">=2.2.0"
accelerate = ">=1.10.0"
datasets = ">=4.0.0"
huggingface-hub = ">=0.34.4"
```

---

## APPENDIX C: File Structure

```
02-ai-engine/
├── heretic_abliteration.py       # Core abliteration engine
├── heretic_optimizer.py          # Optuna optimization
├── heretic_evaluator.py          # Evaluation & metrics
├── heretic_datasets.py           # Dataset management
├── heretic_config.py             # Configuration
├── heretic_mcp_server.py         # MCP server integration
├── heretic_cli.py                # CLI interface
├── heretic_config.toml           # Default configuration
├── HERETIC_INTEGRATION_PLAN.md   # This document
├── HERETIC_TECHNICAL_REPORT.md   # Source repository analysis
│
├── abliterated_models/           # Saved abliterated models
│   ├── uncensored_code_v1/
│   ├── large_v1/
│   └── ...
│
├── refusal_directions/           # Saved refusal direction vectors
│   ├── uncensored_code.pt
│   ├── large.pt
│   └── ...
│
└── optimization_results/         # Optuna study results
    ├── uncensored_code_study.db
    ├── large_study.db
    └── ...
```

---

**END OF INTEGRATION PLAN**

This comprehensive plan provides a complete roadmap for integrating heretic's advanced abliteration techniques into the LAT5150DRVMIL AI Engine, with deep integration into all existing subsystems (DSMIL, TPM, quantum crypto, RAG, memory, self-improvement) and new MCP/CLI interfaces.

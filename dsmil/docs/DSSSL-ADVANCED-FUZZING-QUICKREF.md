# DSSSL Advanced Fuzzing Quick Reference

## Compilation Flags

```bash
# Basic advanced fuzzing
dsmil-clang++ -fsanitize=fuzzer \
               -mllvm -dsssl-coverage \
               -mllvm -dsssl-state-machine \
               -mllvm -dsssl-crypto-metrics \
               -DDSLLVM_ADVANCED_FUZZING=1 \
               harness.cpp \
               -ldsssl_fuzz_telemetry_advanced \
               -o fuzz_advanced

# With performance counters (requires root)
sudo dsmil-clang++ -fsanitize=fuzzer \
                    -mllvm -dsssl-coverage \
                    -DDSLLVM_ENABLE_PERF_COUNTERS=1 \
                    harness.cpp \
                    -ldsssl_fuzz_telemetry_advanced \
                    -o fuzz_perf
```

## Harness Generation

```bash
# Basic harness
dsssl-gen-harness config.yaml harness.cpp

# Advanced harness (grammar + ML + structure-aware)
dsssl-gen-harness config/dsssl_fuzz_telemetry_advanced.yaml \
                  harness_advanced.cpp \
                  --advanced
```

## Runtime Initialization

```c
// Basic
dsssl_fuzz_telemetry_init(NULL, 65536);

// Advanced (with perf counters & ML)
dsssl_fuzz_telemetry_advanced_init(
    "config.yaml",      // Config path
    1048576,            // 1MB ring buffer
    1,                  // Enable perf counters
    1                   // Enable ML
);
```

## Coverage Tracking

```c
// Update coverage map
uint32_t edges[] = {100, 101, 102};
uint32_t states[] = {5, 6};
int new = dsssl_fuzz_update_coverage_map(
    input_hash, edges, 3, states, 2);

// Get statistics
uint32_t total_edges, total_states;
uint64_t unique_inputs;
dsssl_fuzz_get_coverage_stats(
    &total_edges, &total_states, &unique_inputs);
```

## ML Integration

```c
// Get mutation suggestions
dsssl_mutation_metadata_t suggestions[10];
size_t count = dsssl_fuzz_get_mutation_suggestions(
    seed_id, suggestions, 10);

// Compute interestingness
dsssl_coverage_feedback_t feedback = {0};
feedback.new_edges = 5;
double score = dsssl_fuzz_compute_interestingness(
    input_hash, &feedback);
```

## Performance Counters

```c
// Record performance metrics
uint64_t cycles, cache_misses, mispredicts;
read_perf_counters(&cycles, &cache_misses, &mispredicts);
dsssl_fuzz_record_perf_counters(cycles, cache_misses, mispredicts);
```

## Telemetry Export

```c
// Export for ML training
dsssl_fuzz_export_for_ml("training.json", "json");

// Flush with compression
dsssl_fuzz_flush_advanced_events("telemetry.bin.gz", 1);

// Get statistics
uint64_t total;
double rate, utilization;
dsssl_fuzz_get_telemetry_stats(&total, &rate, &utilization);
```

## Configuration Snippets

### Grammar-Based Fuzzing
```yaml
enable_grammar_fuzzing: true
grammar_file: "tls_grammar.bnf"
```

### ML-Guided
```yaml
enable_ml_guided: true
ml_model_path: "models/mutation_model.onnx"
```

### Dictionary
```yaml
enable_dictionary: true
dictionary:
  - "TLS 1.3"
  - "key_share"
```

### Distributed
```yaml
distributed:
  enabled: true
  worker_id: 0
  num_workers: 16
```

### Performance
```yaml
performance:
  enable_parallel: true
  num_threads: 64
  enable_batch: true
  batch_size: 100000
```

## Environment Variables

```bash
# Enable advanced features
export DSLLVM_ADVANCED_FUZZING=1
export DSLLVM_ENABLE_PERF_COUNTERS=1
export DSLLVM_ENABLE_ML=1

# ML model path
export DSLLVM_ML_MODEL_PATH=models/mutation_model.onnx

# Distributed worker
export DSLLVM_WORKER_ID=0
export DSLLVM_NUM_WORKERS=16
```

## Performance Targets

- **1M+ exec/sec** per worker
- **10M+ events/sec** telemetry
- **Sub-μs** coverage updates
- **ms** ML inference
- **GB/s** telemetry export

## Key APIs

| Function | Purpose |
|----------|---------|
| `dsssl_fuzz_telemetry_advanced_init()` | Initialize advanced telemetry |
| `dsssl_fuzz_record_advanced_event()` | Record rich telemetry |
| `dsssl_fuzz_update_coverage_map()` | Update coverage bitmap |
| `dsssl_fuzz_compute_interestingness()` | ML-based scoring |
| `dsssl_fuzz_get_mutation_suggestions()` | ML-guided mutations |
| `dsssl_fuzz_export_for_ml()` | Export training data |
| `dsssl_fuzz_get_telemetry_stats()` | Get statistics |

## File Locations

- **API**: `dsmil/include/dsssl_fuzz_telemetry_advanced.h`
- **Runtime**: `dsmil/runtime/dsssl_fuzz_telemetry_advanced.c`
- **Generator**: `dsmil/tools/dsssl-gen-harness/dsssl-gen-harness-advanced.cpp`
- **Config**: `dsmil/config/dsssl_fuzz_telemetry_advanced.yaml`
- **Guide**: `dsmil/docs/DSSSL-ADVANCED-FUZZING-GUIDE.md`

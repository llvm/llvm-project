# DSSSL Advanced Fuzzing & Telemetry Guide

## Overview

This guide covers the **enhanced** DSSSL fuzzing foundation optimized for next-generation fuzzing techniques and high-performance systems (1+ petaops INT8 capability).

## Enhanced Features

### Advanced Fuzzing Techniques

1. **Grammar-Based Fuzzing** - Structure-aware generation using BNF grammars
2. **ML-Guided Fuzzing** - AI-powered mutation suggestions
3. **Dictionary-Based Fuzzing** - Smart mutation using known patterns
4. **Structure-Aware Fuzzing** - Protocol-aware input generation
5. **Distributed Fuzzing** - Multi-worker coordination
6. **Performance-Optimized** - High-throughput telemetry collection

### Rich Telemetry

- **Performance Counters** - CPU cycles, cache misses, branch mispredictions
- **Coverage Maps** - Fast bitmap-based coverage tracking
- **Mutation Metadata** - Detailed mutation strategy tracking
- **ML Integration** - Interestingness scoring and mutation guidance
- **Batch Processing** - High-throughput event processing

## Quick Start (Advanced)

### 1. Build with Advanced Features

```bash
cmake -DCMAKE_C_COMPILER=dsmil-clang \
      -DCMAKE_CXX_COMPILER=dsmil-clang++ \
      -DDSLLVM_FUZZING=ON \
      -DDSLLVM_TELEMETRY=ON \
      -DDSLLVM_ADVANCED_FUZZING=ON \
      -DDSLLVM_ENABLE_PERF_COUNTERS=ON \
      -DDSLLVM_ENABLE_ML=ON \
      ..
```

### 2. Generate Advanced Harness

```bash
# Generate harness with grammar-based and ML-guided fuzzing
dsssl-gen-harness config/dsssl_fuzz_telemetry_advanced.yaml \
                  harness_advanced.cpp \
                  --advanced
```

### 3. Compile with Advanced Features

```bash
dsmil-clang++ -fsanitize=fuzzer \
               -mllvm -dsssl-coverage \
               -mllvm -dsssl-state-machine \
               -mllvm -dsssl-crypto-metrics \
               -DDSLLVM_ADVANCED_FUZZING=1 \
               harness_advanced.cpp \
               -ldsssl_fuzz_telemetry \
               -ldsssl_fuzz_telemetry_advanced \
               -o fuzz_advanced
```

### 4. Run with Advanced Telemetry

```bash
# Enable performance counters (requires root or perf permissions)
sudo ./fuzz_advanced -runs=10000000 corpus/

# Or run with ML guidance
DSLLVM_ML_MODEL_PATH=models/mutation_model.onnx ./fuzz_advanced corpus/
```

## Advanced Configuration

### YAML Configuration (Advanced)

```yaml
targets:
  tls_advanced:
    type: tls_handshake
    enable_grammar_fuzzing: true
    grammar_file: "tls_grammar.bnf"
    enable_ml_guided: true
    ml_model_path: "models/mutation_model.onnx"
    enable_dictionary: true
    dictionary:
      - "TLS 1.3"
      - "key_share"
      - "supported_versions"
    enable_perf_counters: true
    enable_distributed: true
    worker_id: 0
    num_workers: 16
```

## Grammar-Based Fuzzing

### Grammar File Format (BNF)

```bnf
<tls_handshake> ::= <client_hello> | <server_hello> | <certificate>
<client_hello> ::= "\x01" <version> <random> <session_id> <cipher_suites> <compression> <extensions>
<version> ::= "\x03" "\x03" | "\x03" "\x04"
<cipher_suites> ::= <cipher_suite> | <cipher_suite> <cipher_suites>
```

### Usage

```c
// Grammar-based generation is handled automatically in advanced harness
// The harness parses grammar file and generates structured inputs
```

## ML-Guided Fuzzing

### Model Integration

The advanced harness supports ONNX models for:
- **Mutation Guidance** - Suggests high-value mutations
- **Interestingness Scoring** - Predicts input value
- **Coverage Prediction** - Predicts coverage before execution

### Model Requirements

- **Input**: Mutation metadata, coverage feedback, performance metrics
- **Output**: Mutation suggestions, interestingness scores
- **Format**: ONNX Runtime compatible

### Usage

```c
// Get ML-guided mutation suggestions
dsssl_mutation_metadata_t suggestions[10];
size_t count = dsssl_fuzz_get_mutation_suggestions(seed_id, suggestions, 10);

// Compute interestingness score
double score = dsssl_fuzz_compute_interestingness(input_hash, &feedback);
if (score > 0.7) {
    // High-value input - save to corpus
}
```

## Dictionary-Based Fuzzing

### Dictionary Configuration

```yaml
enable_dictionary: true
dictionary:
  - "TLS 1.3"
  - "TLS 1.2"
  - "GREASE"
  - "key_share"
  - "supported_versions"
  - "signature_algorithms"
```

### Usage

Dictionary entries are automatically inserted into mutations at strategic positions, improving coverage of protocol-specific constructs.

## Structure-Aware Fuzzing

Structure-aware fuzzing understands protocol formats and generates valid structures:

```c
// Automatically parses TLS record structure
uint8_t record_type = data[0];
uint16_t version = (data[1] << 8) | data[2];
uint16_t length = (data[3] << 8) | data[4];

// Validates structure before processing
if (length > max_record_size) return 0;
```

## Distributed Fuzzing

### Configuration

```yaml
distributed:
  enabled: true
  coordinator_url: "http://coordinator:8080"
  worker_id: 0
  num_workers: 16
  sync_interval: 60
  sync_path: "/shared/corpus"
```

### Usage

```bash
# Worker 0
DSLLVM_WORKER_ID=0 ./fuzz_advanced corpus/

# Worker 1
DSLLVM_WORKER_ID=1 ./fuzz_advanced corpus/

# Workers automatically sync corpus via shared path
```

## Performance Counters

### Enabling Performance Counters

Requires Linux `perf` support and appropriate permissions:

```bash
# Run with perf permissions
sudo sysctl -w kernel.perf_event_paranoid=0
sudo ./fuzz_advanced corpus/

# Or use perf directly
perf record -e cycles,cache-misses,branch-misses ./fuzz_advanced corpus/
```

### Metrics Collected

- **CPU Cycles** - Total cycles consumed
- **Cache Misses** - L1/L2/L3 cache misses
- **Branch Mispredictions** - Branch predictor failures
- **TLB Misses** - Translation lookaside buffer misses

## Rich Telemetry Export

### Export Formats

```c
// Export for ML training (JSON)
dsssl_fuzz_export_for_ml("training_data.json", "json");

// Export with compression
dsssl_fuzz_flush_advanced_events("telemetry.bin.gz", 1);
```

### Telemetry Statistics

```c
uint64_t total_events;
double events_per_sec;
double utilization;
dsssl_fuzz_get_telemetry_stats(&total_events, &events_per_sec, &utilization);

printf("Events/sec: %.2f, Buffer utilization: %.2f%%\n", 
       events_per_sec, utilization * 100.0);
```

## Coverage Feedback

### Coverage Map

Fast bitmap-based coverage tracking:

```c
// Update coverage map
uint32_t new_edges[] = {100, 101, 102};
uint32_t new_states[] = {5, 6};
int new_coverage = dsssl_fuzz_update_coverage_map(
    input_hash, new_edges, 3, new_states, 2);

if (new_coverage) {
    // New coverage found - compute interestingness
    dsssl_coverage_feedback_t feedback = {0};
    feedback.new_edges = 3;
    feedback.new_states = 2;
    double score = dsssl_fuzz_compute_interestingness(input_hash, &feedback);
}
```

### Coverage Statistics

```c
uint32_t total_edges, total_states;
uint64_t unique_inputs;
dsssl_fuzz_get_coverage_stats(&total_edges, &total_states, &unique_inputs);

printf("Coverage: %u edges, %u states, %llu unique inputs\n",
       total_edges, total_states, (unsigned long long)unique_inputs);
```

## High-Performance Optimizations

### Batch Processing

Process multiple inputs in batches for better throughput:

```yaml
performance:
  enable_batch: true
  batch_size: 10000
  batch_timeout_ms: 100
```

### Parallel Processing

```yaml
performance:
  enable_parallel: true
  num_threads: 8
  thread_affinity: true
```

### Memory Preallocation

```yaml
performance:
  preallocate_buffers: true
  buffer_size: 1048576
  max_memory_mb: 4096
```

## ML Model Training

### Training Data Collection

```bash
# Run fuzzer in training mode
DSLLVM_ML_TRAINING=1 ./fuzz_advanced corpus/

# Export training data periodically
# Data exported to ml_training_data.json
```

### Model Integration

1. **Train models** using exported telemetry data
2. **Deploy models** as ONNX files
3. **Configure** model paths in YAML
4. **Enable ML** in harness generation

## Best Practices

1. **Use grammar files** for structure-aware fuzzing
2. **Enable performance counters** for side-channel detection
3. **Use ML models** for mutation guidance
4. **Distribute fuzzing** across multiple workers
5. **Export telemetry** regularly for analysis
6. **Monitor coverage stats** to track progress
7. **Use batch processing** for high-throughput scenarios
8. **Preallocate buffers** for consistent performance

## Performance Tuning

### For 1 Petaops Systems

```yaml
performance:
  enable_parallel: true
  num_threads: 64  # Match CPU cores
  enable_batch: true
  batch_size: 100000  # Large batches
  preallocate_buffers: true
  buffer_size: 16777216  # 16MB buffers
  enable_simd: true
  enable_avx512: true
```

### Ring Buffer Sizing

For high-throughput systems:
- **1MB+ ring buffers** for telemetry
- **Large coverage maps** (1M+ entries)
- **Batch event processing**

## Troubleshooting

### Performance Counters Not Working

```bash
# Check permissions
sudo sysctl -w kernel.perf_event_paranoid=0

# Verify perf support
perf list | grep hardware
```

### ML Models Not Loading

1. Verify ONNX Runtime is installed
2. Check model file paths in config
3. Verify model format compatibility

### High Memory Usage

1. Reduce ring buffer size
2. Enable batch processing
3. Reduce coverage map size
4. Flush telemetry more frequently

## See Also

- `dsmil/include/dsssl_fuzz_telemetry_advanced.h` - Advanced API
- `dsmil/config/dsssl_fuzz_telemetry_advanced.yaml` - Advanced config
- `dsmil/docs/DSSSL-FUZZING-GUIDE.md` - Basic fuzzing guide

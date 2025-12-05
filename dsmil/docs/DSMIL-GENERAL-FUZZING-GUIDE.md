# DSLLVM General-Purpose Fuzzing Foundation Guide

## Overview

The DSLLVM General-Purpose Fuzzing Foundation provides a comprehensive, target-agnostic fuzzing infrastructure that can be applied to **any** codebase, not just crypto/TLS. It supports:

- **Any target type**: Protocols, parsers, APIs, libraries, kernels, etc.
- **Advanced techniques**: Grammar-based, ML-guided, structure-aware fuzzing
- **Rich telemetry**: Coverage, performance, security metrics
- **High performance**: Optimized for 1+ petaops systems
- **Distributed fuzzing**: Multi-worker coordination

## Quick Start

### 1. Annotate Your Code

```c
#include "dsmil_fuzz_attributes.h"
#include "dsmil_fuzz_telemetry.h"

// Mark entry point
DSMIL_FUZZ_ENTRY_POINT
DSMIL_FUZZ_COVERAGE
int parse_input(const uint8_t *data, size_t len) {
    // Coverage automatically tracked
    return process(data, len);
}

// Mark state machine
DSMIL_FUZZ_STATE_MACHINE("protocol_state")
int handle_protocol_message(const uint8_t *msg, size_t len) {
    dsmil_fuzz_state_transition(1, 0, 1);
    return 0;
}

// Mark critical operation
DSMIL_FUZZ_CRITICAL_OP("json_parse")
int json_parse(const char *json) {
    dsmil_fuzz_metric_begin("json_parse");
    // ... parsing ...
    dsmil_fuzz_metric_end("json_parse");
    return 0;
}
```

### 2. Generate Harness

```bash
# Create config file
cat > my_target.yaml << EOF
target:
  name: "my_parser"
  type: "parser"
  input_format: "binary"
  max_input_size: 1048576
EOF

# Generate harness
dsmil-gen-fuzz-harness my_target.yaml harness.cpp --advanced
```

### 3. Compile

```bash
dsmil-clang++ -fsanitize=fuzzer \
               -mllvm -dsmil-fuzz-coverage \
               -mllvm -dsmil-fuzz-state-machine \
               harness.cpp \
               your_code.cpp \
               -ldsmil_fuzz_telemetry \
               -o fuzz_target
```

### 4. Run

```bash
./fuzz_target -runs=1000000 corpus/
```

## Target Types

### Generic Target

For any codebase:

```yaml
target:
  name: "my_target"
  type: "generic"
  input_format: "binary"
```

### Protocol Target

For network protocols:

```yaml
target:
  name: "my_protocol"
  type: "protocol"
  input_format: "binary"
  enable_structure_aware: true
  dictionary:
    - "PROTOCOL_MAGIC"
    - "VERSION_1"
```

### Parser Target

For parsers (JSON, XML, etc.):

```yaml
target:
  name: "json_parser"
  type: "parser"
  input_format: "text"
  enable_grammar_fuzzing: true
  grammar_file: "json_grammar.bnf"
```

### API Target

For library APIs:

```yaml
target:
  name: "my_api"
  type: "api"
  input_format: "structured"
  enable_api_misuse: true
```

## Attributes

### Coverage Attributes

```c
DSMIL_FUZZ_COVERAGE              // Enable coverage tracking
DSMIL_FUZZ_ENTRY_POINT           // Mark as primary fuzzing target
```

### State Machine Attributes

```c
DSMIL_FUZZ_STATE_MACHINE("name") // Mark state machine
```

### Operation Attributes

```c
DSMIL_FUZZ_CRITICAL_OP("op_name") // Track operation metrics
DSMIL_FUZZ_CONSTANT_TIME_LOOP     // Mark constant-time loop
```

### API Misuse Attributes

```c
DSMIL_FUZZ_API_MISUSE_CHECK("api_name") // Enable misuse detection
```

## Runtime API

### Basic Telemetry

```c
// Initialize
dsmil_fuzz_telemetry_init("config.yaml", 65536);

// Set context
dsmil_fuzz_set_context(input_hash);

// Coverage
dsmil_fuzz_cov_hit(site_id);

// State machine
dsmil_fuzz_state_transition(sm_id, from, to);

// Metrics
dsmil_fuzz_metric_begin("operation");
dsmil_fuzz_metric_end("operation");

// API misuse
dsmil_fuzz_api_misuse_report("api", "reason", context_id);

// Export
dsmil_fuzz_flush_events("telemetry.bin");
```

### Advanced Telemetry

```c
// Initialize advanced
dsmil_fuzz_telemetry_advanced_init("config.yaml", 1048576, 1, 1);

// Coverage map
dsmil_fuzz_update_coverage_map(input_hash, edges, edge_count, states, state_count);

// ML integration
double score = dsmil_fuzz_compute_interestingness(input_hash, &feedback);
size_t count = dsmil_fuzz_get_mutation_suggestions(seed_id, suggestions, max);

// Performance
dsmil_fuzz_record_perf_counters(cycles, cache_misses, mispredicts);

// Export for ML
dsmil_fuzz_export_for_ml("training.json", "json");
```

## Example Use Cases

### HTTP Parser Fuzzing

```c
DSMIL_FUZZ_STATE_MACHINE("http_parser")
DSMIL_FUZZ_COVERAGE
int http_parse(const uint8_t *data, size_t len) {
    // Parser implementation
    return 0;
}
```

Config:
```yaml
target:
  name: "http_parser"
  type: "parser"
  enable_dictionary: true
  dictionary: ["GET", "POST", "HTTP/1.1"]
```

### JSON Parser Fuzzing

```c
DSMIL_FUZZ_CRITICAL_OP("json_parse")
DSMIL_FUZZ_COVERAGE
int json_parse(const char *json) {
    dsmil_fuzz_metric_begin("json_parse");
    // Parse JSON
    dsmil_fuzz_metric_end("json_parse");
    return 0;
}
```

### Network Protocol Fuzzing

```c
DSMIL_FUZZ_STATE_MACHINE("protocol_sm")
DSMIL_FUZZ_COVERAGE
int process_protocol_message(const uint8_t *msg, size_t len) {
    dsmil_fuzz_state_transition(1, 0, 1);
    // Process message
    return 0;
}
```

### File Format Puzzing

```c
DSMIL_FUZZ_ENTRY_POINT
DSMIL_FUZZ_COVERAGE
int parse_file_format(const uint8_t *data, size_t len) {
    // Parse file format
    return 0;
}
```

## Compiler Flags

| Flag | Description |
|------|-------------|
| `-mllvm -dsmil-fuzz-coverage` | Enable coverage instrumentation |
| `-mllvm -dsmil-fuzz-state-machine` | Enable state machine tracking |
| `-mllvm -dsmil-fuzz-metrics` | Enable operation metrics |
| `-mllvm -dsmil-fuzz-api-misuse` | Enable API misuse detection |

## Integration Examples

### libFuzzer

```bash
dsmil-clang++ -fsanitize=fuzzer \
               -mllvm -dsmil-fuzz-coverage \
               harness.cpp target.cpp \
               -ldsmil_fuzz_telemetry \
               -o fuzz_target
```

### AFL++

```bash
afl-clang++ -mllvm -dsmil-fuzz-coverage \
            harness.cpp target.cpp \
            -ldsmil_fuzz_telemetry \
            -o fuzz_target
```

### Custom Fuzzer

```c
#include "dsmil_fuzz_telemetry.h"

void custom_fuzzer_loop(void) {
    dsmil_fuzz_telemetry_init(NULL, 65536);
    
    while (1) {
        uint8_t *input = get_next_input();
        size_t len = get_input_size();
        
        uint64_t hash = compute_hash(input, len);
        dsmil_fuzz_set_context(hash);
        
        target_function(input, len);
        
        dsmil_fuzz_clear_events();
    }
}
```

## Best Practices

1. **Mark all entry points** with `DSMIL_FUZZ_ENTRY_POINT`
2. **Annotate state machines** for protocol/parser targets
3. **Use critical op markers** for performance-sensitive code
4. **Enable API misuse checks** for security-critical APIs
5. **Configure budgets** for constant-time operations
6. **Export telemetry** regularly for analysis
7. **Use advanced features** for high-value targets

## See Also

- `dsmil/include/dsmil_fuzz_telemetry.h` - Basic API
- `dsmil/include/dsmil_fuzz_telemetry_advanced.h` - Advanced API
- `dsmil/include/dsmil_fuzz_attributes.h` - Attributes
- `dsmil/examples/generic_fuzz_example.c` - Examples
- `dsmil/docs/DSSSL-ADVANCED-FUZZING-GUIDE.md` - Advanced features

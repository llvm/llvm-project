# DSLLVM General-Purpose Fuzzing Quick Reference

## Compilation

```bash
# Basic fuzzing
dsmil-clang++ -fsanitize=fuzzer \
               -mllvm -dsmil-fuzz-coverage \
               harness.cpp target.cpp \
               -ldsmil_fuzz_telemetry \
               -o fuzz_target

# Advanced fuzzing
dsmil-clang++ -fsanitize=fuzzer \
               -mllvm -dsmil-fuzz-coverage \
               -mllvm -dsmil-fuzz-state-machine \
               -DDSMIL_ADVANCED_FUZZING=1 \
               harness.cpp target.cpp \
               -ldsmil_fuzz_telemetry \
               -ldsmil_fuzz_telemetry_advanced \
               -o fuzz_target
```

## Harness Generation

```bash
# Generate harness
dsmil-gen-fuzz-harness config.yaml harness.cpp

# Advanced harness
dsmil-gen-fuzz-harness config.yaml harness.cpp --advanced
```

## Attributes

```c
DSMIL_FUZZ_COVERAGE                    // Coverage tracking
DSMIL_FUZZ_ENTRY_POINT                 // Primary target
DSMIL_FUZZ_STATE_MACHINE("name")       // State machine
DSMIL_FUZZ_CRITICAL_OP("op")           // Operation metrics
DSMIL_FUZZ_API_MISUSE_CHECK("api")     // Misuse detection
DSMIL_FUZZ_CONSTANT_TIME_LOOP           // Constant-time loop
```

## Runtime API

```c
// Initialize
dsmil_fuzz_telemetry_init("config.yaml", 65536);

// Context
dsmil_fuzz_set_context(input_hash);

// Coverage
dsmil_fuzz_cov_hit(site_id);

// State machine
dsmil_fuzz_state_transition(sm_id, from, to);

// Metrics
dsmil_fuzz_metric_begin("operation");
dsmil_fuzz_metric_end("operation");

// Export
dsmil_fuzz_flush_events("telemetry.bin");
```

## Config Template

```yaml
target:
  name: "my_target"
  type: "generic"  # generic, protocol, parser, api
  input_format: "binary"  # binary, text, structured
  max_input_size: 1048576
  enable_grammar_fuzzing: false
  enable_ml_guided: false
  enable_dictionary: false
```

## Target Types

- **generic** - Any codebase
- **protocol** - Network protocols
- **parser** - Parsers (JSON, XML, etc.)
- **api** - Library APIs

## Files

- **API**: `dsmil/include/dsmil_fuzz_telemetry.h`
- **Advanced**: `dsmil/include/dsmil_fuzz_telemetry_advanced.h`
- **Attributes**: `dsmil/include/dsmil_fuzz_attributes.h`
- **Generator**: `dsmil/tools/dsmil-gen-fuzz-harness/dsmil-gen-fuzz-harness.cpp`
- **Config**: `dsmil/config/fuzz_telemetry_generic.yaml`

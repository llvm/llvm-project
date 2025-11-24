# DSLLVM Fuzz Harness Schema Specification

**Version:** 1.3.0
**Feature:** Auto-Generated Fuzz & Chaos Harnesses (Phase 1)
**Schema ID:** `dsmil-fuzz-v1`
**SPDX-License-Identifier:** Apache-2.0 WITH LLVM-exception

## Overview

The DSLLVM fuzz harness schema (`.dsmilfuzz.json`) provides a machine-readable specification of fuzz targets extracted from compiled code. These specifications can be consumed by:

- **Fuzzing Engines:** libFuzzer, AFL++, Honggfuzz, etc.
- **AI Harness Generators:** Layer 7 LLM for automated harness code generation
- **CI/CD Pipelines:** Automated continuous fuzzing integration
- **Layer 8 Security AI:** Risk assessment and chaos scenario generation

## Schema Format

### Top-Level Structure

```json
{
  "schema": "dsmil-fuzz-v1",
  "version": "1.3.0",
  "binary": "<module_name>",
  "generated_at": "<ISO 8601 timestamp>",
  "compiler_version": "<DSLLVM version>",
  "fuzz_targets": [ ... ],
  "l7_llm_integration": { ... },
  "l8_chaos_scenarios": [ ... ]
}
```

### Fields

#### `schema` (string, required)

Schema identifier. Always `"dsmil-fuzz-v1"` for this version.

#### `version` (string, required)

DSLLVM version that generated this file. Format: `"MAJOR.MINOR.PATCH"`.

#### `binary` (string, required)

Name of the binary/module being fuzzed.

#### `generated_at` (string, required)

ISO 8601 timestamp of schema generation.

**Example:** `"2026-01-15T14:30:00Z"`

#### `compiler_version` (string, optional)

Full DSLLVM compiler version string.

**Example:** `"DSLLVM 1.3.0-dev (based on LLVM 18.0.0)"`

#### `fuzz_targets` (array, required)

Array of fuzz target objects. See [Fuzz Target Object](#fuzz-target-object).

#### `l7_llm_integration` (object, optional)

Layer 7 LLM integration metadata. See [L7 LLM Integration](#l7-llm-integration).

#### `l8_chaos_scenarios` (array, optional)

Layer 8 Security AI chaos testing scenarios. See [L8 Chaos Scenarios](#l8-chaos-scenarios).

## Fuzz Target Object

Each fuzz target describes a function with untrusted input that should be fuzzed.

```json
{
  "function": "<function_name>",
  "untrusted_params": [ "<param1>", "<param2>" ],
  "parameter_domains": { ... },
  "l8_risk_score": 0.87,
  "priority": "high",
  "layer": 8,
  "device": 80,
  "stage": "serve",
  "call_graph_depth": 5,
  "complexity_score": 0.65
}
```

### Fields

#### `function` (string, required)

Fully qualified function name (with namespace/module prefix if applicable).

**Example:** `"parse_network_packet"`, `"MyNamespace::decode_message"`

#### `untrusted_params` (array of strings, required)

List of parameter names that ingest untrusted data.

**Example:** `["packet_data", "length"]`

#### `parameter_domains` (object, required)

Map of parameter name → parameter domain specification. See [Parameter Domain](#parameter-domain-object).

#### `l8_risk_score` (float, required)

Layer 8 Security AI risk score (0.0 = no risk, 1.0 = critical risk).

Computed based on:
- Function complexity
- Number of untrusted parameters
- Pointer/buffer operations
- Call graph depth
- Layer assignment (lower layers = higher privilege)
- Historical vulnerability patterns

**Example:** `0.87` (high risk)

#### `priority` (string, required)

Human-readable priority level derived from risk score.

**Values:** `"high"`, `"medium"`, `"low"`

**Mapping:**
- `risk >= 0.7` → `"high"`
- `risk >= 0.4` → `"medium"`
- `risk < 0.4` → `"low"`

#### `layer` (integer, optional)

DSMIL layer assignment (0-8). Lower layers indicate higher privilege and security criticality.

**Example:** `8` (Security AI layer)

#### `device` (integer, optional)

DSMIL device assignment (0-103).

**Example:** `80` (Security AI device)

#### `stage` (string, optional)

MLOps stage annotation.

**Values:** `"pretrain"`, `"finetune"`, `"quantized"`, `"distilled"`, `"serve"`, `"debug"`, `"experimental"`

#### `call_graph_depth` (integer, optional)

Maximum call depth from this function (complexity metric).

#### `complexity_score` (float, optional)

Normalized cyclomatic complexity (0.0-1.0).

## Parameter Domain Object

Describes the valid domain for a fuzz target parameter.

```json
{
  "type": "bytes",
  "length_ref": "length",
  "min": 0,
  "max": 65535,
  "constraints": [ ... ]
}
```

### Fields

#### `type` (string, required)

Parameter type category.

**Supported Types:**

| Type | Description | Example C Type |
|------|-------------|----------------|
| `bytes` | Byte buffer | `uint8_t*`, `char*` |
| `int8_t` | 8-bit signed integer | `int8_t` |
| `int16_t` | 16-bit signed integer | `int16_t` |
| `int32_t` | 32-bit signed integer | `int32_t` |
| `int64_t` | 64-bit signed integer | `int64_t` |
| `uint8_t` | 8-bit unsigned integer | `uint8_t` |
| `uint16_t` | 16-bit unsigned integer | `uint16_t` |
| `uint32_t` | 32-bit unsigned integer | `uint32_t` |
| `uint64_t` | 64-bit unsigned integer | `uint64_t` |
| `float` | 32-bit floating-point | `float` |
| `double` | 64-bit floating-point | `double` |
| `struct` | Structured type | `struct foo` |
| `array` | Fixed-size array | `int[10]` |
| `unknown` | Unknown/opaque type | `void*` |

#### `length_ref` (string, optional)

For `bytes` type: name of parameter that specifies the buffer length.

**Example:** If function is `parse(uint8_t *buf, size_t len)`, then:
```json
{
  "buf": {
    "type": "bytes",
    "length_ref": "len"
  }
}
```

#### `min` (integer/float, optional)

Minimum valid value for numeric types.

**Example:** `0` (non-negative integers), `-100` (signed integers)

#### `max` (integer/float, optional)

Maximum valid value for numeric types.

**Example:** `65535` (16-bit limit), `1048576` (1MB buffer limit)

#### `constraints` (array of strings, optional)

Additional constraints in human-readable form.

**Examples:**
- `"must be null-terminated"`
- `"must be aligned to 16 bytes"`
- `"must start with magic number 0x89504E47"`

## L7 LLM Integration

Metadata for Layer 7 LLM harness code generation.

```json
{
  "enabled": true,
  "request_harness_generation": true,
  "target_fuzzer": "libFuzzer",
  "output_language": "C++",
  "harness_template": "dsmil_libfuzzer_v1",
  "l7_service_url": "http://layer7-llm.local:8080/api/v1/generate"
}
```

### Fields

#### `enabled` (boolean, required)

Whether L7 LLM integration is enabled.

#### `request_harness_generation` (boolean, optional)

If true, requests L7 LLM to generate full harness code.

#### `target_fuzzer` (string, optional)

Target fuzzing engine.

**Supported:** `"libFuzzer"`, `"AFL++"`, `"Honggfuzz"`, `"custom"`

#### `output_language` (string, optional)

Language for generated harness code.

**Supported:** `"C"`, `"C++"`, `"Rust"`

#### `harness_template` (string, optional)

Template ID for harness generation.

**Standard Templates:**
- `"dsmil_libfuzzer_v1"` - Standard libFuzzer harness
- `"dsmil_afl_v1"` - AFL++ harness with shared memory
- `"dsmil_chaos_v1"` - Chaos testing harness (fault injection)

#### `l7_service_url` (string, optional)

URL of Layer 7 LLM service for harness generation.

## L8 Chaos Scenarios

Layer 8 Security AI chaos testing scenarios for advanced fuzzing.

```json
{
  "scenario_id": "memory_pressure",
  "description": "Test under extreme memory pressure",
  "fault_injection": {
    "malloc_failure_rate": 0.1,
    "oom_trigger_threshold": "90%"
  },
  "target_functions": ["parse_network_packet"],
  "expected_behavior": "graceful_degradation"
}
```

### Fields

#### `scenario_id` (string, required)

Unique identifier for chaos scenario.

**Standard Scenarios:**
- `"memory_pressure"` - OOM conditions
- `"network_latency"` - High latency/packet loss
- `"disk_full"` - Full filesystem
- `"race_conditions"` - Thread interleaving
- `"signal_injection"` - Unexpected signals
- `"corrupted_input"` - Bit flips in input data

#### `description` (string, required)

Human-readable description of scenario.

#### `fault_injection` (object, optional)

Fault injection parameters specific to scenario.

#### `target_functions` (array of strings, optional)

List of functions to apply chaos scenario to. If empty, applies to all fuzz targets.

#### `expected_behavior` (string, required)

Expected behavior under chaos conditions.

**Values:**
- `"graceful_degradation"` - Function should return error, not crash
- `"no_corruption"` - State remains consistent
- `"bounded_resource_use"` - Resource usage stays within limits
- `"crash_safe"` - Process can crash but no memory corruption

## Complete Example

### Example 1: Network Packet Parser

**Function:**
```c
DSMIL_UNTRUSTED_INPUT
DSMIL_LAYER(8)
DSMIL_DEVICE(80)
void parse_network_packet(const uint8_t *packet_data, size_t length);
```

**Generated `.dsmilfuzz.json`:**
```json
{
  "schema": "dsmil-fuzz-v1",
  "version": "1.3.0",
  "binary": "network_daemon",
  "generated_at": "2026-01-15T14:30:00Z",
  "compiler_version": "DSLLVM 1.3.0-dev",
  "fuzz_targets": [
    {
      "function": "parse_network_packet",
      "untrusted_params": ["packet_data", "length"],
      "parameter_domains": {
        "packet_data": {
          "type": "bytes",
          "length_ref": "length",
          "constraints": ["must be valid Ethernet frame"]
        },
        "length": {
          "type": "uint64_t",
          "min": 0,
          "max": 65535,
          "constraints": ["must match actual packet size"]
        }
      },
      "l8_risk_score": 0.87,
      "priority": "high",
      "layer": 8,
      "device": 80,
      "stage": "serve",
      "call_graph_depth": 5,
      "complexity_score": 0.72
    }
  ],
  "l7_llm_integration": {
    "enabled": true,
    "request_harness_generation": true,
    "target_fuzzer": "libFuzzer",
    "output_language": "C++",
    "harness_template": "dsmil_libfuzzer_v1"
  },
  "l8_chaos_scenarios": [
    {
      "scenario_id": "corrupted_input",
      "description": "Test with bit-flipped network packets",
      "fault_injection": {
        "bit_flip_rate": 0.001,
        "byte_corruption_rate": 0.01
      },
      "target_functions": ["parse_network_packet"],
      "expected_behavior": "graceful_degradation"
    },
    {
      "scenario_id": "oversized_packets",
      "description": "Test with packets exceeding MTU",
      "fault_injection": {
        "length_multiplier": 10,
        "max_size": 655350
      },
      "target_functions": ["parse_network_packet"],
      "expected_behavior": "no_corruption"
    }
  ]
}
```

### Example 2: JSON Parser

**Function:**
```c
DSMIL_UNTRUSTED_INPUT
DSMIL_LAYER(7)
int parse_json(const char *json_str, size_t len, struct json_object *out);
```

**Generated `.dsmilfuzz.json`:**
```json
{
  "schema": "dsmil-fuzz-v1",
  "version": "1.3.0",
  "binary": "api_server",
  "generated_at": "2026-01-15T14:35:00Z",
  "fuzz_targets": [
    {
      "function": "parse_json",
      "untrusted_params": ["json_str", "len"],
      "parameter_domains": {
        "json_str": {
          "type": "bytes",
          "length_ref": "len",
          "constraints": [
            "UTF-8 encoded",
            "may contain embedded nulls"
          ]
        },
        "len": {
          "type": "uint64_t",
          "min": 0,
          "max": 1048576,
          "constraints": ["max 1MB JSON document"]
        },
        "out": {
          "type": "struct",
          "constraints": ["pointer must be valid"]
        }
      },
      "l8_risk_score": 0.65,
      "priority": "medium",
      "layer": 7,
      "stage": "serve"
    }
  ],
  "l7_llm_integration": {
    "enabled": true,
    "request_harness_generation": true,
    "target_fuzzer": "libFuzzer",
    "output_language": "C++",
    "harness_template": "dsmil_libfuzzer_v1"
  }
}
```

## Consuming the Schema

### Fuzzing Engine Integration

#### libFuzzer Harness Generation

```bash
# Generate libFuzzer harness using L7 LLM
dsmil-fuzz-gen network_daemon.dsmilfuzz.json --fuzzer=libFuzzer

# Output: network_daemon_fuzz.cpp
```

**Generated Harness Example:**
```cpp
#include <cstdint>
#include <cstddef>

// Forward declaration
extern "C" void parse_network_packet(const uint8_t *packet_data, size_t length);

// libFuzzer entry point
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    // Enforce length constraints from parameter_domains
    if (size > 65535) return 0;  // max from schema

    // Call fuzz target
    parse_network_packet(data, size);

    return 0;
}
```

#### AFL++ Integration

```bash
# Generate AFL++ harness
dsmil-fuzz-gen network_daemon.dsmilfuzz.json --fuzzer=AFL++

# Compile with AFL++
afl-clang-fast++ -o network_daemon_fuzz network_daemon_fuzz.cpp network_daemon.o

# Run fuzzer
afl-fuzz -i seeds -o findings -- ./network_daemon_fuzz @@
```

### CI/CD Integration

```yaml
# .gitlab-ci.yml example
fuzz_network_daemon:
  stage: security
  script:
    # Compile with fuzz export enabled
    - dsmil-clang -fdsmil-fuzz-export -fdsmil-fuzz-l7-llm src/network.c -o network_daemon

    # Generate harnesses using L7 LLM
    - dsmil-fuzz-gen network_daemon.dsmilfuzz.json --fuzzer=libFuzzer

    # Compile fuzz harnesses
    - clang++ -fsanitize=fuzzer,address network_daemon_fuzz.cpp -o fuzz_harness

    # Run fuzzer for 1 hour
    - timeout 3600 ./fuzz_harness -max_total_time=3600 -print_final_stats=1

  artifacts:
    paths:
      - "*.dsmilfuzz.json"
      - crash-*
      - leak-*
```

### Layer 8 Chaos Testing

```bash
# Run chaos testing scenarios
dsmil-chaos-test network_daemon.dsmilfuzz.json --scenario=all

# Output:
# [Scenario: corrupted_input] PASS (10000 iterations, 0 crashes)
# [Scenario: oversized_packets] PASS (10000 iterations, 0 crashes)
# [Scenario: memory_pressure] FAIL (crashed after 532 iterations)
```

## Schema Versioning

### Version History

- **v1.0** (DSLLVM 1.3.0): Initial release
  - Basic fuzz target specification
  - L7 LLM integration
  - L8 chaos scenarios

### Future Versions

- **v2.0** (planned): Add support for stateful fuzzing, corpus minimization hints

## References

- **Fuzz Export Pass:** `dsmil/lib/Passes/DsmilFuzzExportPass.cpp`
- **Attributes Header:** `dsmil/include/dsmil_attributes.h`
- **DSLLVM Roadmap:** `dsmil/docs/DSLLVM-ROADMAP.md`
- **libFuzzer:** https://llvm.org/docs/LibFuzzer.html
- **AFL++:** https://github.com/AFLplusplus/AFLplusplus

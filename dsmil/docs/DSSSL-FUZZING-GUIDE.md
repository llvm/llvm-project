# DSSSL Advanced Fuzzing & Telemetry Guide

## Overview

The DSSSL Fuzzing & Telemetry Extension provides comprehensive fuzzing support for hardened OpenSSL forks (DSSSL). It enables:

1. **Coverage-guided fuzzing** with edge coverage and state machine tracking
2. **Side-channel detection** via crypto operation metrics
3. **API misuse detection** for common security issues
4. **Rich telemetry** for offline ML analysis and CI gating
5. **Automated harness generation** for TLS, X.509, and state machine fuzzing

## Architecture

### Components

1. **Instrumentation Passes** (LLVM)
   - `DssslCoveragePass` - Coverage and state machine instrumentation
   - `DssslCryptoMetricsPass` - Crypto operation metrics
   - `DssslApiMisusePass` - API misuse detection wrappers

2. **Runtime Library** (`libdsssl_fuzz_telemetry.a`)
   - Telemetry collection and ring buffer management
   - Budget enforcement
   - Event export

3. **Harness Generator** (`dsssl-gen-harness`)
   - Generates libFuzzer/AFL++ harnesses from YAML configs
   - Supports TLS, X.509, and state machine targets

## Quick Start

### 1. Build DSSSL with Fuzzing Support

```bash
# Configure with DSLLVM
cmake -DCMAKE_C_COMPILER=dsmil-clang \
      -DCMAKE_CXX_COMPILER=dsmil-clang++ \
      -DDSLLVM_FUZZING=ON \
      -DDSLLVM_TELEMETRY=ON \
      -DDSLLVM_CRYPTO_BUDGETS_CONFIG=/path/to/dsssl_fuzz_telemetry.yaml \
      ..

# Build
make
```

### 2. Annotate Your Code

```c
#include "dsssl_fuzz_attributes.h"
#include "dsssl_fuzz_telemetry.h"

// Mark state machine function
DSSSL_STATE_MACHINE("tls_handshake")
DSSSL_COVERAGE
int tls_process_handshake(SSL *ssl, const uint8_t *data, size_t len) {
    // Coverage and state transitions automatically tracked
    return 0;
}

// Mark crypto function
DSSSL_CRYPTO("ecdsa_sign")
int ecdsa_sign(const EC_KEY *key, uint8_t *sig, size_t *sig_len,
               const uint8_t *msg, size_t msg_len) {
    dsssl_crypto_metric_begin("ecdsa_sign");
    // ... signing operation ...
    dsssl_crypto_metric_end("ecdsa_sign");
    return 0;
}
```

### 3. Generate Fuzz Harness

```bash
# Generate TLS dialect harness
dsssl-gen-harness config/tls_dialect_config.yaml harness_tls.cpp

# Generate X.509 PKI harness
dsssl-gen-harness config/x509_pki_config.yaml harness_x509.cpp

# Generate state machine harness
dsssl-gen-harness config/tls_state_config.yaml harness_state.cpp
```

### 4. Compile Harness

```bash
# With libFuzzer
dsmil-clang++ -fsanitize=fuzzer \
               -mllvm -dsssl-coverage \
               -mllvm -dsssl-state-machine \
               -mllvm -dsssl-crypto-metrics \
               harness_tls.cpp \
               -ldsssl_fuzz_telemetry \
               -o fuzz_tls

# With AFL++
afl-clang++ -mllvm -dsssl-coverage \
            -mllvm -dsssl-state-machine \
            harness_tls.cpp \
            -ldsssl_fuzz_telemetry \
            -o fuzz_tls
```

### 5. Run Fuzzer

```bash
# libFuzzer
./fuzz_tls -runs=1000000 corpus/

# AFL++
afl-fuzz -i input/ -o output/ ./fuzz_tls @@
```

## Instrumentation Passes

### Coverage & State Machine Pass

**Flag**: `-mllvm -dsssl-coverage` and `-mllvm -dsssl-state-machine`

**What it does**:
- Instruments functions marked with `DSSSL_COVERAGE` or `DSSSL_STATE_MACHINE`
- Adds coverage counters for edges (branches)
- Tracks state machine transitions

**Attributes**:
```c
DSSSL_COVERAGE                    // Enable coverage instrumentation
DSSSL_STATE_MACHINE("tls_handshake")  // Mark state machine
```

**Runtime API**:
```c
void dsssl_cov_hit(uint32_t site_id);
void dsssl_state_transition(uint16_t sm_id, uint16_t state_from, uint16_t state_to);
```

### Crypto Metrics Pass

**Flag**: `-mllvm -dsssl-crypto-metrics`

**What it does**:
- Instruments functions marked with `DSSSL_CRYPTO`
- Tracks branch counts, load/store counts
- Optionally measures timing (with `-mllvm -dsssl-crypto-timing`)

**Attributes**:
```c
DSSSL_CRYPTO("ecdsa_sign")        // Mark crypto operation
DSSSL_CONSTANT_TIME_LOOP          // Mark constant-time critical loop
```

**Runtime API**:
```c
void dsssl_crypto_metric_begin(const char *op_name);
void dsssl_crypto_metric_end(const char *op_name);
void dsssl_crypto_metric_record(const char *op_name, uint32_t branches,
                                uint32_t loads, uint32_t stores, uint64_t cycles);
int dsssl_crypto_check_budget(const char *op_name, uint32_t branches,
                             uint32_t loads, uint32_t stores, uint64_t cycles);
```

### API Misuse Pass

**Flag**: `-mllvm -dsssl-api-misuse`

**What it does**:
- Wraps critical API calls with misuse detection
- Checks for nonce reuse, bad IVs, disabled cert checks, etc.

**Attributes**:
```c
DSSSL_API_MISUSE_CHECK("AEAD_init")  // Enable misuse checks
```

**Runtime API**:
```c
void dsssl_api_misuse_report(const char *api, const char *reason, uint64_t context_id);
```

## Configuration

### YAML Configuration File

Create `dsssl_fuzz_telemetry.yaml`:

```yaml
# Crypto operation budgets
crypto_budgets:
  ecdsa_sign:
    max_branches: 5000
    max_loads: 20000
    max_stores: 10000
    max_delta_cycles: 2000

# Fuzzing targets
targets:
  tls_dialect:
    type: tls_handshake
    role: client
    use_0rtt: true
    use_tickets: true

# API misuse policies
api_misuse_policies:
  AEAD_init:
    check_nonce_length: true
    check_nonce_reuse: true
    abort_on_violation: false

# Telemetry settings
telemetry:
  ring_buffer_size: 65536
  flush_on_exit: true
  enable_timing: false
```

## Fuzzing Targets

### TLS Dialect Fuzzing

Fuzzes TLS handshake variations, cipher suites, extensions, and protocol quirks.

**Config** (`tls_dialect_config.yaml`):
```yaml
targets:
  tls_dialect:
    type: tls_handshake
    role: client
    max_record_size: 16384
    use_0rtt: true
    use_tickets: true
    use_psk: true
```

**Generate harness**:
```bash
dsssl-gen-harness config/tls_dialect_config.yaml harness_tls.cpp
```

**What it fuzzes**:
- TLS version negotiation
- Cipher suite lists
- Extensions (ALPN, SNI, key_share, etc.)
- 0-RTT data
- Session tickets
- PSK bindings

### X.509 PKI Path Fuzzing

Fuzzes certificate chain validation, path building, and name constraints.

**Config** (`x509_pki_config.yaml`):
```yaml
targets:
  x509_pki:
    type: x509_path
    max_chain_len: 8
    fuzz_name_constraints: true
    fuzz_idn: true
```

**Generate harness**:
```bash
dsssl-gen-harness config/x509_pki_config.yaml harness_x509.cpp
```

**What it fuzzes**:
- ASN.1 DER certificate structures
- Certificate chain construction
- Path validation logic
- Name constraints
- Internationalized Domain Names (IDN)

### TLS State Machine Fuzzing

Fuzzes session state, tickets, PSKs, and 0-RTT acceptance.

**Config** (`tls_state_config.yaml`):
```yaml
targets:
  tls_state:
    type: tls_state_machine
    fuzz_tickets: true
    fuzz_psk_binding: true
    fuzz_0rtt: true
```

**Generate harness**:
```bash
dsssl-gen-harness config/tls_state_config.yaml harness_state.cpp
```

**What it fuzzes**:
- Ticket issuance and usage
- PSK identity binding
- 0-RTT acceptance rules
- Replay counters
- State confusion scenarios

## Telemetry Collection

### Event Types

- `DSSSL_EVENT_COVERAGE_HIT` - Coverage site hit
- `DSSSL_EVENT_STATE_TRANSITION` - State machine transition
- `DSSSL_EVENT_CRYPTO_METRIC` - Crypto operation metrics
- `DSSSL_EVENT_API_MISUSE` - API misuse detected
- `DSSSL_EVENT_PKI_DECISION` - PKI validation decision
- `DSSSL_EVENT_TICKET_EVENT` - Ticket issue/use/expire
- `DSSSL_EVENT_BUDGET_VIOLATION` - Budget violation

### Exporting Telemetry

```c
// Initialize telemetry
dsssl_fuzz_telemetry_init("dsssl_fuzz_telemetry.yaml", 65536);

// Set context ID (hash of fuzz input)
dsssl_fuzz_set_context(hash_of_input);

// ... run fuzzed code ...

// Flush events to file
dsssl_fuzz_flush_events("telemetry.bin");

// Or retrieve events programmatically
dsssl_telemetry_event_t events[1024];
size_t count = dsssl_fuzz_get_events(events, 1024);
```

### Analyzing Telemetry

Telemetry events can be analyzed offline for:
- Coverage analysis
- State machine exploration
- Side-channel detection
- API misuse patterns
- Budget violations

## Budget Enforcement

### Crypto Budgets

Budgets are defined in YAML config:

```yaml
crypto_budgets:
  ecdsa_sign:
    max_branches: 5000
    max_loads: 20000
    max_stores: 10000
    max_delta_cycles: 2000
```

When a budget is violated:
1. A `DSSSL_EVENT_BUDGET_VIOLATION` event is recorded
2. Optionally, `abort()` is called (in fuzz builds) to surface as crash

### State Machine Budgets

```yaml
state_machine_budgets:
  tls_handshake:
    max_transitions: 50
```

## Integration with libFuzzer

### Basic Integration

```cpp
#include "dsssl_fuzz_telemetry.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    // Initialize telemetry (once)
    static bool initialized = false;
    if (!initialized) {
        dsssl_fuzz_telemetry_init(NULL, 65536);
        initialized = true;
    }
    
    // Set context ID
    uint64_t context_id = hash_input(data, size);
    dsssl_fuzz_set_context(context_id);
    
    // Run fuzzed code
    process_tls_handshake(data, size);
    
    return 0;
}
```

### Compilation

```bash
dsmil-clang++ -fsanitize=fuzzer \
               -mllvm -dsssl-coverage \
               -mllvm -dsssl-state-machine \
               -mllvm -dsssl-crypto-metrics \
               harness.cpp \
               -ldsssl_fuzz_telemetry \
               -o fuzz_target
```

## Integration with AFL++

### Basic Integration

Same harness code works with AFL++:

```bash
afl-clang++ -mllvm -dsssl-coverage \
            -mllvm -dsssl-state-machine \
            harness.cpp \
            -ldsssl_fuzz_telemetry \
            -o fuzz_target
```

### Running AFL++

```bash
afl-fuzz -i input/ -o output/ ./fuzz_target @@
```

## CMake Integration

### CMakeLists.txt Example

```cmake
# Enable DSLLVM fuzzing
set(CMAKE_C_COMPILER "dsmil-clang")
set(CMAKE_CXX_COMPILER "dsmil-clang++")

# Add fuzzing flags
add_compile_options(
    -mllvm -dsssl-coverage
    -mllvm -dsssl-state-machine
    -mllvm -dsssl-crypto-metrics
    -mllvm -dsssl-api-misuse
)

# Link telemetry library
target_link_libraries(your_target
    dsssl_fuzz_telemetry
)

# Fuzzing build type
if(DSLLVM_FUZZING)
    add_compile_definitions(DSLLVM_FUZZING=1)
    add_compile_options(-fsanitize=fuzzer)
endif()
```

## Best Practices

1. **Annotate all critical functions** with appropriate attributes
2. **Use state machine annotations** for protocol state tracking
3. **Set crypto budgets** based on constant-time requirements
4. **Enable API misuse checks** for security-critical APIs
5. **Export telemetry** for offline analysis
6. **Use generated harnesses** as starting points
7. **Customize harnesses** for specific test scenarios
8. **Monitor budget violations** for side-channel detection

## Troubleshooting

### Telemetry Not Appearing

1. Check initialization: `dsssl_fuzz_telemetry_init()` called?
2. Verify context ID is set: `dsssl_fuzz_set_context()`
3. Check ring buffer size is sufficient
4. Ensure events are flushed: `dsssl_fuzz_flush_events()`

### Budget Violations Not Detected

1. Verify budgets are loaded from YAML config
2. Check `dsssl_crypto_check_budget()` is called
3. Ensure metrics are recorded: `dsssl_crypto_metric_record()`

### Coverage Not Working

1. Verify pass is enabled: `-mllvm -dsssl-coverage`
2. Check functions are annotated: `DSSSL_COVERAGE`
3. Ensure runtime library is linked: `-ldsssl_fuzz_telemetry`

## Example Workflow

```bash
# 1. Configure
cp dsmil/config/dsssl_fuzz_telemetry.yaml .

# 2. Annotate code
# Add DSSSL_* attributes to your DSSSL code

# 3. Generate harness
dsssl-gen-harness config/tls_dialect_config.yaml harness.cpp

# 4. Compile
dsmil-clang++ -fsanitize=fuzzer \
               -mllvm -dsssl-coverage \
               -mllvm -dsssl-state-machine \
               harness.cpp \
               -ldsssl_fuzz_telemetry \
               -o fuzz_tls

# 5. Run
./fuzz_tls -runs=1000000 corpus/

# 6. Analyze telemetry
# Process telemetry.bin for coverage, budgets, etc.
```

## See Also

- `dsmil/include/dsssl_fuzz_telemetry.h` - Runtime API
- `dsmil/include/dsssl_fuzz_attributes.h` - Attribute macros
- `dsmil/examples/dsssl_fuzz_example.c` - Complete example
- `dsmil/config/dsssl_fuzz_telemetry.yaml` - Configuration template

# DSMIL Test Suite

This directory contains comprehensive tests for DSLLVM functionality.

## Test Categories

### Layer Policy Tests (`dsmil/layer_policies/`)

Test enforcement of DSMIL layer boundary policies.

**Test Cases**:
- ✅ Same-layer calls (should pass)
- ✅ Downward calls (higher → lower layer, should pass)
- ❌ Upward calls without gateway (should fail)
- ✅ Upward calls with gateway (should pass)
- ❌ Clearance violations (should fail)
- ✅ Clearance with gateway (should pass)
- ❌ ROE escalation without gateway (should fail)

**Example Test**:
```c
// RUN: dsmil-clang -fpass-pipeline=dsmil-default %s -o /dev/null 2>&1 | FileCheck %s

#include <dsmil_attributes.h>

DSMIL_LAYER(1)
void kernel_operation(void) { }

DSMIL_LAYER(7)
void user_function(void) {
    // CHECK: error: layer boundary violation
    // CHECK: caller 'user_function' (layer 7) calls 'kernel_operation' (layer 1) without dsmil_gateway
    kernel_operation();
}
```

**Run Tests**:
```bash
ninja -C build check-dsmil-layer
```

---

### Stage Policy Tests (`dsmil/stage_policies/`)

Test MLOps stage policy enforcement.

**Test Cases**:
- ✅ Production with `serve` stage (should pass)
- ❌ Production with `debug` stage (should fail)
- ❌ Production with `experimental` stage (should fail)
- ✅ Production with `quantized` stage (should pass)
- ❌ Layer ≥3 with `pretrain` stage (should fail)
- ✅ Development with any stage (should pass)

**Example Test**:
```c
// RUN: env DSMIL_POLICY=production dsmil-clang -fpass-pipeline=dsmil-default %s -o /dev/null 2>&1 | FileCheck %s

#include <dsmil_attributes.h>

// CHECK: error: stage policy violation
// CHECK: production binaries cannot link dsmil_stage("debug") code
DSMIL_STAGE("debug")
void debug_diagnostics(void) { }

DSMIL_STAGE("serve")
int main(void) {
    debug_diagnostics();
    return 0;
}
```

**Run Tests**:
```bash
ninja -C build check-dsmil-stage
```

---

### Provenance Tests (`dsmil/provenance/`)

Test CNSA 2.0 provenance generation and verification.

**Test Cases**:

**Generation**:
- ✅ Basic provenance record creation
- ✅ SHA-384 hash computation
- ✅ ML-DSA-87 signature generation
- ✅ ELF section embedding
- ✅ Encrypted provenance with ML-KEM-1024
- ✅ Certificate chain embedding

**Verification**:
- ✅ Valid signature verification
- ❌ Invalid signature (should fail)
- ❌ Tampered binary (hash mismatch, should fail)
- ❌ Expired certificate (should fail)
- ❌ Revoked key (should fail)
- ✅ Encrypted provenance decryption

**Example Test**:
```bash
#!/bin/bash
# RUN: %s %t

# Generate test keys
dsmil-keygen --type psk --test --output $TMPDIR/test_psk.pem

# Compile with provenance
export DSMIL_PSK_PATH=$TMPDIR/test_psk.pem
dsmil-clang -fpass-pipeline=dsmil-default -o %t/binary test_input.c

# Verify provenance
dsmil-verify %t/binary
# CHECK: ✓ Provenance present
# CHECK: ✓ Signature valid

# Tamper with binary
echo "tampered" >> %t/binary

# Verification should fail
dsmil-verify %t/binary
# CHECK: ✗ Binary hash mismatch
```

**Run Tests**:
```bash
ninja -C build check-dsmil-provenance
```

---

### Sandbox Tests (`dsmil/sandbox/`)

Test sandbox wrapper injection and enforcement.

**Test Cases**:

**Wrapper Generation**:
- ✅ `main` renamed to `main_real`
- ✅ New `main` injected with sandbox setup
- ✅ Profile loaded correctly
- ✅ Capabilities dropped
- ✅ Seccomp filter installed

**Runtime**:
- ✅ Allowed syscalls succeed
- ❌ Disallowed syscalls blocked by seccomp
- ❌ Privilege escalation attempts fail
- ✅ Resource limits enforced

**Example Test**:
```c
// RUN: dsmil-clang -fpass-pipeline=dsmil-default %s -o %t/binary -ldsmil_sandbox_runtime
// RUN: %t/binary
// RUN: dmesg | grep dsmil | FileCheck %s

#include <dsmil_attributes.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>

DSMIL_SANDBOX("l7_llm_worker")
int main(void) {
    // CHECK: DSMIL: Sandbox 'l7_llm_worker' applied

    // Allowed operation
    printf("Hello from sandbox\n");

    // Disallowed operation (should be blocked by seccomp)
    // This will cause SIGSYS and program termination
    // CHECK: DSMIL: Seccomp violation: socket (syscall 41)
    socket(AF_INET, SOCK_STREAM, 0);

    return 0;
}
```

**Run Tests**:
```bash
ninja -C build check-dsmil-sandbox
```

---

## Test Infrastructure

### LIT Configuration

Tests use LLVM's LIT (LLVM Integrated Tester) framework.

**Configuration**: `test/dsmil/lit.cfg.py`

**Test Formats**:
- `.c` / `.cpp`: C/C++ source files with embedded RUN/CHECK directives
- `.ll`: LLVM IR files
- `.sh`: Shell scripts for integration tests

### FileCheck

Tests use LLVM's FileCheck for output verification:

```c
// RUN: dsmil-clang %s -o /dev/null 2>&1 | FileCheck %s
// CHECK: error: layer boundary violation
// CHECK-NEXT: note: caller 'foo' is at layer 7
```

**FileCheck Directives**:
- `CHECK`: Match pattern
- `CHECK-NEXT`: Match on next line
- `CHECK-NOT`: Pattern must not appear
- `CHECK-DAG`: Match in any order

---

## Running Tests

### All DSMIL Tests

```bash
ninja -C build check-dsmil
```

### Specific Test Categories

```bash
ninja -C build check-dsmil-layer      # Layer policy tests
ninja -C build check-dsmil-stage      # Stage policy tests
ninja -C build check-dsmil-provenance # Provenance tests
ninja -C build check-dsmil-sandbox    # Sandbox tests
```

### Individual Tests

```bash
# Run specific test
llvm-lit test/dsmil/layer_policies/upward-call-no-gateway.c -v

# Run with filter
llvm-lit test/dsmil -v --filter="layer"
```

### Debug Failed Tests

```bash
# Show full output
llvm-lit test/dsmil/layer_policies/upward-call-no-gateway.c -v -a

# Keep temporary files
llvm-lit test/dsmil -v --no-execute
```

---

## Test Coverage

### Current Coverage Goals

- **Pass Tests**: 100% line coverage for all DSMIL passes
- **Runtime Tests**: 100% line coverage for runtime libraries
- **Integration Tests**: End-to-end scenarios for all pipelines
- **Security Tests**: Negative tests for all security features

### Measuring Coverage

```bash
# Build with coverage
cmake -G Ninja -S llvm -B build \
  -DLLVM_ENABLE_DSMIL=ON \
  -DLLVM_BUILD_INSTRUMENTED_COVERAGE=ON

# Run tests
ninja -C build check-dsmil

# Generate report
llvm-cov show build/bin/dsmil-clang \
  -instr-profile=build/profiles/default.profdata \
  -output-dir=coverage-report
```

---

## Writing Tests

### Test File Template

```c
// RUN: dsmil-clang -fpass-pipeline=dsmil-default %s -o /dev/null 2>&1 | FileCheck %s
// REQUIRES: dsmil

#include <dsmil_attributes.h>

// Test description: Verify that ...

DSMIL_LAYER(7)
void test_function(void) {
    // Test code
}

// CHECK: expected output
// CHECK-NOT: unexpected output

int main(void) {
    test_function();
    return 0;
}
```

### Best Practices

1. **One Test, One Feature**: Each test should focus on a single feature or edge case
2. **Clear Naming**: Use descriptive test file names (e.g., `upward-call-with-gateway.c`)
3. **Comment Test Intent**: Add `// Test description:` at the top
4. **Check All Output**: Verify both positive and negative cases
5. **Use FileCheck Patterns**: Make checks robust with regex where needed

---

## Implementation Status

### Layer Policy Tests
- [ ] Same-layer calls
- [ ] Downward calls
- [ ] Upward calls without gateway
- [ ] Upward calls with gateway
- [ ] Clearance violations
- [ ] ROE escalation

### Stage Policy Tests
- [ ] Production enforcement
- [ ] Development flexibility
- [ ] Layer-stage interactions

### Provenance Tests
- [ ] Generation
- [ ] Signing
- [ ] Verification
- [ ] Encrypted provenance
- [ ] Tampering detection

### Sandbox Tests
- [ ] Wrapper injection
- [ ] Capability enforcement
- [ ] Seccomp enforcement
- [ ] Resource limits

---

## Contributing

When adding tests:

1. Follow the test file template
2. Add both positive and negative test cases
3. Use meaningful CHECK patterns
4. Test edge cases and error paths
5. Update CMakeLists.txt to include new tests

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for details.

---

## Continuous Integration

Tests run automatically on:

- **Pre-commit**: Fast smoke tests (~2 min)
- **Pull Request**: Full test suite (~15 min)
- **Nightly**: Extended tests + fuzzing + sanitizers (~2 hours)

**CI Configuration**: `.github/workflows/dsmil-tests.yml`

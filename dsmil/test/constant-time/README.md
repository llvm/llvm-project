# DSMIL Constant-Time Enforcement Tests

This directory contains test cases for DSLLVM's constant-time enforcement feature, which prevents timing side-channel attacks in cryptographic code.

## Overview

The `dsmil-ct-check` pass enforces that functions marked with `DSMIL_SECRET` attribute execute in constant time, preventing attackers from learning secret information through timing measurements.

## Test Categories

### Good Examples (Should Pass)

- **good_constant_time.c**: Demonstrates correct constant-time cryptographic primitives
  - Constant-time memory comparison
  - Constant-time conditional select using arithmetic
  - Constant-time table lookups using masking
  - Constant-time swaps

### Violation Examples (Should Fail)

- **bad_secret_branch.c**: Secret-dependent branches
  - Early-exit comparisons
  - Switch statements on secrets
  - If-else on secret-derived conditions

- **bad_secret_memory.c**: Secret-dependent memory accesses
  - Array indexing with secret values (cache timing leaks)
  - Table lookups with secret indices
  - Secret-dependent pointer arithmetic

- **bad_variable_time.c**: Variable-time instructions
  - Division/modulo with secret operands
  - Variable shifts with secret amounts
  - Data-dependent loop iterations

## Violation Types

The `dsmil-ct-check` pass detects four categories of violations:

1. **SECRET_BRANCH**: Branching on secret-derived conditions
   - `if`, `switch`, `select` (ternary `?:`)
   - Loop conditions dependent on secrets

2. **SECRET_MEMORY**: Secret-dependent memory access patterns
   - Array indexing: `array[secret_index]`
   - Computed addresses using secrets
   - Cache timing side-channels

3. **VARIABLE_TIME**: Instructions with data-dependent timing
   - Division: `secret / value` or `secret % value`
   - Variable shifts: `value << secret_count`
   - Data-dependent iterations

4. **SECRET_LEAK**: Unintentional secret exposure
   - Non-secret functions returning secret values
   - Storing secrets to non-secret memory

## Running Tests

### Compile Good Example (Should Succeed)

```bash
dsmil-clang -O2 -fdsmil-ct-check \
    good_constant_time.c -o good_ct.o

# Expected output:
# [DSMIL Constant-Time] No violations found
```

### Compile Bad Examples (Should Fail)

```bash
dsmil-clang -O2 -fdsmil-ct-check -fdsmil-ct-check-strict \
    bad_secret_branch.c -o bad_branch.o

# Expected output:
# [DSMIL Constant-Time] Found N violations:
#   [SECRET_BRANCH] bad_memcmp_early_exit:XX
#     Secret-dependent branch: branching on secret value...
# FATAL ERROR: Constant-time violations detected in strict mode
```

## Constant-Time Programming Guidelines

### ✅ DO

- Use bitwise operations instead of branches:
  ```c
  // Good: constant-time select
  uint32_t mask = -(condition != 0);
  result = (a & mask) | (b & ~mask);
  ```

- Use masking for table lookups:
  ```c
  // Good: access all entries, mask the one you want
  for (size_t i = 0; i < size; i++) {
    uint8_t mask = -((uint8_t)(i == index));
    result |= table[i] & mask;
  }
  ```

- Iterate full lengths without early exit:
  ```c
  // Good: always scan full length
  for (size_t i = 0; i < len; i++) {
    result |= a[i] ^ b[i];  // accumulate differences
  }
  ```

### ❌ DON'T

- Branch on secret values:
  ```c
  // Bad: timing reveals comparison result
  if (secret_key[0] == guess) {
    return 1;
  }
  ```

- Index arrays with secrets:
  ```c
  // Bad: cache timing leaks index
  output = sbox[secret_byte];
  ```

- Use division/modulo on secrets:
  ```c
  // Bad: timing depends on operands
  remainder = secret % divisor;
  ```

## Integration with DSMIL Architecture

Constant-time enforcement integrates with:

- **Layer 8 Security AI**: Analyzes constant-time functions for side-channel vulnerabilities
- **Layer 5 Performance AI**: Balances constant-time enforcement with performance
- **CNSA 2.0 Crypto**: Required for all ML-KEM, ML-DSA key material
- **Mission Profiles**: `border_ops` requires strict constant-time for all crypto

## References

- [DSLLVM Design Specification](../../docs/DSLLVM-DESIGN.md#104-constant-time--side-channel-annotations-dsmil_secret)
- [DSMIL Attributes Reference](../../docs/ATTRIBUTES.md)
- [Security Boulevard: Constant-Time Support Lands in LLVM](https://securityboulevard.com/2025/11/constant-time-support-lands-in-llvm-protecting-cryptographic-code-at-the-compiler-level/)

## Additional Resources

### Academic Papers

- "Cache-Timing Attacks on AES" (Bernstein, 2005)
- "Constant-Time Implementations of Cryptographic Primitives" (CHES 2016)
- "FaCT: A Flexible, Constant-Time Programming Language" (PLDI 2017)

### Real-World Examples

See `tpm2_compat/` for constant-time implementations of:
- ML-KEM-1024 (Kyber) encapsulation
- ML-DSA-87 (Dilithium) signing
- AES-256-GCM authenticated encryption
- SHA-384 hashing

All cryptographic code in DSMIL Layers 8-9 must use `DSMIL_SECRET` and pass `dsmil-ct-check`.

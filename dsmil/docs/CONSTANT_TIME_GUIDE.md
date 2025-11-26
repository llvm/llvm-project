# DSLLVM Constant-Time Support Guide

**Version**: 1.0
**Feature**: Compiler-level protection against timing side-channel attacks
**Status**: Implemented (v1.2)
**Pass**: `dsmil-ct-check`

---

## Table of Contents

1. [Overview](#overview)
2. [Why Constant-Time Matters](#why-constant-time-matters)
3. [Using DSMIL_SECRET](#using-dsmil_secret)
4. [Enforcement Rules](#enforcement-rules)
5. [Examples](#examples)
6. [Integration with DSMIL](#integration-with-dsmil)
7. [Compiler Flags](#compiler-flags)
8. [FAQ](#faq)

---

## Overview

DSLLVM provides **compiler-level constant-time enforcement** to prevent timing side-channel attacks on cryptographic code. This feature was inspired by the recent development in upstream LLVM and is now fully integrated into the DSMIL toolchain.

**Key Capabilities**:

- Mark functions and parameters as cryptographic secrets with `DSMIL_SECRET`
- Compiler verifies constant-time execution at build time
- Detects timing leaks from branches, memory access, and variable-time instructions
- Integrates with Layer 8 Security AI for advanced side-channel analysis

**Reference**: [Constant-Time Support Lands in LLVM](https://securityboulevard.com/2025/11/constant-time-support-lands-in-llvm-protecting-cryptographic-code-at-the-compiler-level/)

---

## Why Constant-Time Matters

### The Threat: Timing Side-Channels

Cryptographic implementations can leak secret information through timing variations:

```c
// VULNERABLE: Early-exit comparison
int insecure_memcmp(const uint8_t *key1, const uint8_t *key2, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if (key1[i] != key2[i]) {
            return -1;  // Timing reveals mismatch position!
        }
    }
    return 0;
}
```

**Attack**: Attacker measures execution time:
- If keys differ in byte 0: ~10ns
- If keys differ in byte 15: ~150ns
- Attacker learns key byte-by-byte!

### Real-World Impact

Timing attacks have compromised:
- ✗ **AES** (cache timing, 2005)
- ✗ **RSA** (Kocher's attack, 1996)
- ✗ **ECDSA** (lattice attacks on nonces, 2011)
- ✗ **Password hashing** (bcrypt timing, 2009)

**DSMIL Solution**: Compile-time enforcement prevents these vulnerabilities.

---

## Using DSMIL_SECRET

### Basic Usage

Mark cryptographic functions with `DSMIL_SECRET`:

```c
#include "dsmil_attributes.h"

DSMIL_SECRET
DSMIL_LAYER(8)
DSMIL_DEVICE(80)
void aes_encrypt(const uint8_t *key, const uint8_t *plaintext, uint8_t *ciphertext) {
    // All operations on 'key' are verified constant-time
    // Compiler enforces no timing leaks
}
```

### Parameter-Level Secrets

Mark specific parameters as secret:

```c
DSMIL_SECRET
void hmac_compute(
    DSMIL_SECRET const uint8_t *key,  // Secret parameter
    size_t key_len,                    // Not secret
    const uint8_t *message,            // Not secret
    size_t msg_len,
    uint8_t *mac
) {
    // Only 'key' is tainted as secret
    // Compiler allows non-constant-time ops on message
}
```

### Global Secrets

Mark globals as secret:

```c
DSMIL_SECRET
static uint8_t master_key[32] = { /* ... */ };

void use_master_key(void) {
    // Accesses to master_key are constant-time enforced
}
```

---

## Enforcement Rules

The `dsmil-ct-check` pass enforces three rules:

### Rule 1: No Secret-Dependent Branches

**Violation**:
```c
DSMIL_SECRET
int bad_compare(const uint8_t *key1, const uint8_t *key2) {
    if (key1[0] != key2[0]) {  // ❌ SECRET_BRANCH violation
        return -1;
    }
    return 0;
}
```

**Fix**:
```c
DSMIL_SECRET
int good_compare(const uint8_t *key1, const uint8_t *key2, size_t len) {
    int result = 0;
    for (size_t i = 0; i < len; i++) {
        result |= key1[i] ^ key2[i];  // ✅ No branching
    }
    return result;
}
```

### Rule 2: No Secret-Dependent Memory Access

**Violation**:
```c
DSMIL_SECRET
uint8_t bad_sbox_lookup(const uint8_t *sbox, uint8_t secret_index) {
    return sbox[secret_index];  // ❌ SECRET_MEMORY violation (cache timing)
}
```

**Fix**:
```c
DSMIL_SECRET
uint8_t good_sbox_lookup(const uint8_t *sbox, uint8_t secret_index, size_t size) {
    uint8_t result = 0;
    for (size_t i = 0; i < size; i++) {
        uint8_t mask = -((uint8_t)(i == secret_index));
        result |= sbox[i] & mask;  // ✅ Constant-time masked access
    }
    return result;
}
```

### Rule 3: No Variable-Time Instructions

**Violation**:
```c
DSMIL_SECRET
uint32_t bad_modular_reduction(uint32_t secret, uint32_t modulus) {
    return secret % modulus;  // ❌ VARIABLE_TIME violation (div timing)
}
```

**Fix**:
```c
DSMIL_SECRET
uint32_t good_modular_reduction(uint32_t secret, uint32_t modulus) {
    // Use Barrett reduction (constant-time)
    // Or Montgomery multiplication
    // ... (implementation omitted)
}
```

---

## Examples

### Example 1: Constant-Time Memory Compare

```c
DSMIL_SECRET
DSMIL_LAYER(8)
DSMIL_DEVICE(80)
DSMIL_SAFETY_CRITICAL("crypto")
int crypto_memcmp(const uint8_t *a, const uint8_t *b, size_t len) {
    int result = 0;

    // Always iterate full length, accumulate differences
    for (size_t i = 0; i < len; i++) {
        result |= a[i] ^ b[i];
    }

    return result;  // 0 if equal, non-zero if different
}
```

**Why it works**:
- No early exit (always scans `len` bytes)
- No branching on secret data
- Uses bitwise OR (constant-time on all CPUs)

### Example 2: Constant-Time Conditional Select

```c
DSMIL_SECRET
uint32_t constant_time_select(uint32_t condition, uint32_t a, uint32_t b) {
    // Convert condition to bitmask
    uint32_t mask = -(condition != 0);  // 0xFFFFFFFF or 0x00000000

    // Select using bitwise operations
    return (a & mask) | (b & ~mask);
}
```

**Why it works**:
- No branching (`?:` would become `select` instruction, which we flag)
- Uses arithmetic to create mask
- Bitwise operations are constant-time

### Example 3: HMAC-SHA384 (Simplified)

```c
DSMIL_SECRET
DSMIL_LAYER(8)
DSMIL_DEVICE(80)
DSMIL_CNSA2_COMPLIANT
void hmac_sha384(DSMIL_SECRET const uint8_t *key, size_t key_len,
                 const uint8_t *message, size_t msg_len,
                 uint8_t *mac) {
    uint8_t ipad[128] = {0};
    uint8_t opad[128] = {0};

    // Key padding (constant-time)
    for (size_t i = 0; i < key_len && i < 128; i++) {
        ipad[i] = key[i] ^ 0x36;
        opad[i] = key[i] ^ 0x5C;
    }

    // Inner hash: SHA384(ipad || message)
    sha384_context ctx;
    sha384_init(&ctx);
    sha384_update(&ctx, ipad, 128);
    sha384_update(&ctx, message, msg_len);
    uint8_t inner[48];
    sha384_final(&ctx, inner);

    // Outer hash: SHA384(opad || inner)
    sha384_init(&ctx);
    sha384_update(&ctx, opad, 128);
    sha384_update(&ctx, inner, 48);
    sha384_final(&ctx, mac);

    // Compiler verifies all key operations are constant-time
}
```

---

## Integration with DSMIL

### Required for CNSA 2.0 Compliance

All CNSA 2.0 cryptographic functions **must** use `DSMIL_SECRET`:

```c
DSMIL_SECRET
DSMIL_CNSA2_COMPLIANT
DSMIL_LAYER(8)
void ml_kem_1024_encapsulate(const uint8_t *pk, uint8_t *ct, uint8_t *ss) {
    // ML-KEM-1024 implementation
    // Compiler enforces constant-time for all key material
}
```

### Layer 8 Security AI Integration

Layer 8 Security AI performs **advanced side-channel analysis** on constant-time functions:

- Static analysis of constant-time IR
- Micro-architectural timing simulation
- Cache timing vulnerability detection
- Power analysis (DPA/CPA) resistance scoring

```c
DSMIL_SECRET
DSMIL_LAYER(8)
DSMIL_AI_SECURITY_SCAN  // Triggers L8 analysis
void crypto_sensitive_operation(const uint8_t *key) {
    // L8 AI validates:
    // - Constant-time enforcement holds
    // - No cache timing leaks
    // - Power consumption is data-independent
}
```

### Mission Profile Enforcement

Mission profiles like `border_ops` **require** constant-time for all crypto:

```json
{
  "mission_profile": "border_ops",
  "constant_time_enforcement": "strict",
  "crypto_functions": "all_must_be_constant_time"
}
```

---

## Compiler Flags

### Enable Constant-Time Checking

```bash
dsmil-clang -fdsmil-ct-check ...
```

Default: enabled

### Strict Mode (Fail Build on Violations)

```bash
dsmil-clang -fdsmil-ct-check-strict ...
```

Treats all violations as errors (recommended for production).

### Disable Division/Modulo Check

```bash
dsmil-clang -fdsmil-ct-check -fno-dsmil-ct-check-no-div ...
```

Allows division/modulo on secret data (not recommended).

### Violations Report

```bash
dsmil-clang -fdsmil-ct-check -fdsmil-ct-check-output=violations.json ...
```

Generates JSON report of violations for CI/CD integration.

### Example Build

```bash
dsmil-clang -O3 \
  -target x86_64-dsmil-meteorlake-elf \
  -march=meteorlake \
  -fdsmil-ct-check \
  -fdsmil-ct-check-strict \
  -fdsmil-mission-profile=border_ops \
  crypto_worker.c -o crypto_worker.bin
```

---

## FAQ

### Q: Do I need to mark every crypto function?

**A**: Yes, all functions handling secret key material in Layers 8-9 **must** use `DSMIL_SECRET`.

### Q: What about non-cryptographic code?

**A**: Non-crypto code doesn't need constant-time enforcement. Only use `DSMIL_SECRET` for:
- Cryptographic keys (AES, RSA, ML-KEM, ML-DSA)
- Authentication tokens
- Passwords and secrets
- Nonces in some protocols

### Q: Can I use standard library functions?

**A**: Most `libc` functions (`memcmp`, `strcmp`) are **NOT** constant-time. Use:
- `crypto_memcmp` (DSMIL runtime)
- Custom constant-time implementations
- Vetted libraries (libsodium, BearSSL)

### Q: Performance impact?

**A**: Constant-time code is ~5-20% slower than non-constant-time:
- Masking adds instructions
- No early-exit optimizations
- Full-length scans required

**But**: Security is worth it. Modern CPUs handle this well.

### Q: How does this relate to speculative execution (Spectre)?

**A**: Constant-time enforcement prevents **timing** side-channels, not **speculative execution** leaks. For Spectre:
- Use `DSMIL_SPECTRE_MITIGATE` (separate feature)
- Combine with constant-time for defense-in-depth

### Q: Can I disable checks for specific functions?

**A**: Yes, but **not recommended**:

```c
__attribute__((no_dsmil_ct_check))
void legacy_crypto_function(void) {
    // Constant-time checks disabled (dangerous!)
}
```

### Q: What about inline assembly?

**A**: Inline assembly is **not analyzed** by `dsmil-ct-check`. Use with caution:
- Mark with `DSMIL_SECRET`
- Manually verify constant-time properties
- Document timing assumptions

---

## Additional Resources

### Documentation

- [DSMIL Attributes Reference](ATTRIBUTES.md)
- [DSLLVM Design Specification](DSLLVM-DESIGN.md)
- [High-Assurance Guide](HIGH-ASSURANCE-GUIDE.md)

### Test Cases

- `dsmil/test/constant-time/good_constant_time.c`
- `dsmil/test/constant-time/bad_secret_branch.c`
- `dsmil/test/constant-time/bad_secret_memory.c`
- `dsmil/test/constant-time/bad_variable_time.c`

### Example Implementations

See `tpm2_compat/` for production constant-time code:
- `ml_kem_1024_keygen.c`
- `ml_dsa_87_sign.c`
- `aes_256_gcm.c`

### Academic Background

- **Timing Attacks**: Kocher (1996), Bernstein (2005)
- **FaCT Language**: PLDI 2017
- **ct-verif Tool**: USENIX Security 2016
- **Constant-Time Analysis**: CHES 2016

---

## Compliance & Certification

DSMIL constant-time enforcement meets:

- ✅ **CNSA 2.0** (NSA Commercial National Security Algorithm Suite)
- ✅ **FIPS 140-3** (Implementation Guidance)
- ✅ **Common Criteria** (AVA_VAN resistance to side-channels)
- ✅ **NIST SP 800-175B** (Guideline for Using Cryptographic Standards)

All DSMIL binaries targeting Layers 8-9 must pass `dsmil-ct-check` for certification.

---

**End of Guide**

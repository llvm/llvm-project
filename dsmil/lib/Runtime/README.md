# DSMIL Runtime Libraries

This directory contains runtime support libraries linked into DSMIL binaries.

## Libraries

### `libdsmil_sandbox_runtime.a`

Runtime support for sandbox setup and enforcement.

**Dependencies**:
- libcap-ng (capability management)
- libseccomp (seccomp-bpf filter installation)

**Functions**:
- `dsmil_load_sandbox_profile()`: Load sandbox profile (uses dynamic path resolution)
- `dsmil_apply_sandbox()`: Apply sandbox to current process
- `dsmil_apply_capabilities()`: Set capability bounding set
- `dsmil_apply_seccomp()`: Install seccomp BPF filter
- `dsmil_apply_resource_limits()`: Set rlimits

**Path Resolution**: Sandbox profiles are loaded from `${DSMIL_CONFIG_DIR}/sandbox/` (default: `/etc/dsmil/sandbox/`). See [PATH-CONFIGURATION.md](../../docs/PATH-CONFIGURATION.md) for details.

**Used By**: Binaries compiled with `dsmil_sandbox` attribute (via `DsmilSandboxWrapPass`)

**Build**:
```bash
ninja -C build dsmil_sandbox_runtime
```

**Link**:
```bash
dsmil-clang -o binary input.c -ldsmil_sandbox_runtime -lcap-ng -lseccomp
```

---

### `libdsmil_provenance_runtime.a`

Runtime support for provenance generation, verification, and extraction.

**Dependencies**:
- **libdsssl (REQUIRED)** - DSMIL-Grade OpenSSL for SHA-384 and cryptographic operations
- liboqs (Open Quantum Safe) for ML-DSA-87, ML-KEM-1024
- libcbor (CBOR encoding/decoding)
- libelf (ELF binary manipulation)

**Note**: [DSSSL](https://github.com/SWORDIntel/DSSSL) is **required** for DSLLVM. It is a hardened OpenSSL 3.x fork providing enhanced security, PQC support, and TPM integration. OpenSSL 3.x fallback is available for development only.

**Functions**:

**Build-Time** (used by `DsmilProvenancePass`):
- `dsmil_build_provenance()`: Collect metadata and construct provenance record
- `dsmil_sign_provenance()`: Sign with ML-DSA-87 using PSK
- `dsmil_encrypt_sign_provenance()`: Encrypt with ML-KEM-1024 + sign
- `dsmil_embed_provenance()`: Embed in ELF `.note.dsmil.provenance` section

**Runtime** (used by `dsmil-verify`, kernel LSM):
- `dsmil_extract_provenance()`: Extract from ELF binary
- `dsmil_verify_provenance()`: Verify signature and certificate chain
- `dsmil_verify_binary_hash()`: Recompute and verify binary hash
- `dsmil_extract_encrypted_provenance()`: Decrypt + verify

**Utilities**:
- `dsmil_get_build_timestamp()`: ISO 8601 timestamp
- `dsmil_get_git_info()`: Extract Git metadata
- `dsmil_hash_file_sha384()`: Compute file hash

**Build**:
```bash
ninja -C build dsmil_provenance_runtime
```

**Link**:
```bash
# With DSSSL (REQUIRED for production)
dsmil-clang -o binary input.c -ldsmil_provenance_runtime -loqs -lcbor -lelf -ldsssl

# Development/testing with OpenSSL 3.x fallback (not recommended)
dsmil-clang -o binary input.c -ldsmil_provenance_runtime -loqs -lcbor -lelf -lcrypto
```

---

## Directory Structure

```
Runtime/
├── dsmil_sandbox_runtime.c       # Sandbox runtime implementation
├── dsmil_provenance_runtime.c    # Provenance runtime implementation
├── dsmil_paths_runtime.c        # Dynamic path resolution (v1.6.1+) ⭐ NEW
├── dsmil_crypto.c                # CNSA 2.0 crypto wrappers
├── dsmil_elf.c                   # ELF manipulation utilities
└── CMakeLists.txt                # Build configuration
```

## CNSA 2.0 Cryptographic Support

### Algorithms

| Algorithm | Library | Purpose |
|-----------|---------|---------|
| SHA-384 | DSSSL/OpenSSL 3.x | Hashing |
| ML-DSA-87 | liboqs (DSSSL integrated) | Digital signatures (FIPS 204) |
| ML-KEM-1024 | liboqs (DSSSL integrated) | Key encapsulation (FIPS 203) |
| AES-256-GCM | DSSSL/OpenSSL 3.x | AEAD encryption |

### Constant-Time Operations

All cryptographic operations use constant-time implementations to prevent side-channel attacks:

- ML-DSA/ML-KEM: liboqs constant-time implementations
- SHA-384: Hardware-accelerated (Intel SHA Extensions) when available
- AES-256-GCM: AES-NI instructions

### FIPS 140-3 Compliance

Target configuration:
- Use FIPS-validated libcrypto
- liboqs will be FIPS 140-3 validated (post-FIPS 203/204 approval)
- Hardware RNG (RDRAND/RDSEED) for key generation

---

## Sandbox Profiles

Predefined sandbox profiles in `${DSMIL_CONFIG_DIR}/sandbox/` (default: `/etc/dsmil/sandbox/`):

### `l7_llm_worker.profile`

Layer 7 LLM inference worker:

```json
{
  "name": "l7_llm_worker",
  "description": "LLM inference worker with minimal privileges",
  "capabilities": [],
  "syscalls": [
    "read", "write", "mmap", "munmap", "brk",
    "futex", "exit", "exit_group", "rt_sigreturn",
    "clock_gettime", "gettimeofday"
  ],
  "network": {
    "allow": false
  },
  "filesystem": {
    "allowed_paths": ["/opt/dsmil/models"],
    "readonly": true
  },
  "limits": {
    "max_memory_bytes": 17179869184,
    "max_cpu_time_sec": 3600,
    "max_open_files": 256
  }
}
```

### `l5_network_daemon.profile`

Layer 5 network service:

```json
{
  "name": "l5_network_daemon",
  "description": "Network daemon with limited privileges",
  "capabilities": ["CAP_NET_BIND_SERVICE"],
  "syscalls": [
    "read", "write", "socket", "bind", "listen",
    "accept", "connect", "sendto", "recvfrom",
    "mmap", "munmap", "brk", "futex", "exit"
  ],
  "network": {
    "allow": true,
    "allowed_ports": [80, 443, 8080]
  },
  "filesystem": {
    "allowed_paths": ["/etc", "/var/run"],
    "readonly": false
  },
  "limits": {
    "max_memory_bytes": 4294967296,
    "max_cpu_time_sec": 86400,
    "max_open_files": 1024
  }
}
```

---

## Testing

Runtime libraries have comprehensive unit tests:

```bash
# All runtime tests
ninja -C build check-dsmil-runtime

# Sandbox tests
ninja -C build check-dsmil-sandbox

# Provenance tests
ninja -C build check-dsmil-provenance
```

### Manual Testing

```bash
# Test sandbox setup
./test-sandbox l7_llm_worker

# Test provenance generation
./test-provenance-generate /tmp/test_binary

# Test provenance verification
./test-provenance-verify /tmp/test_binary
```

---

### `libdsmil_paths_runtime.a` ⭐ NEW

Runtime support for dynamic path resolution (v1.6.1+).

**Dependencies**: None (pure C, standard library only)

**Functions**:
- `dsmil_get_prefix()`: Get installation prefix
- `dsmil_get_config_dir()`: Get configuration directory
- `dsmil_get_bin_dir()`: Get binary directory
- `dsmil_get_truststore_dir()`: Get truststore directory
- `dsmil_resolve_config()`: Resolve configuration file paths
- `dsmil_resolve_binary()`: Resolve binary paths
- `dsmil_path_exists()`: Check if path exists
- `dsmil_ensure_dir()`: Create directory tree

**Used By**: All DSMIL tools and runtime libraries for portable installations

**Build**:
```bash
ninja -C build dsmil_paths_runtime
```

**Link**:
```bash
dsmil-clang -o binary input.c -ldsmil_paths_runtime
```

**Documentation**: See [PATH-CONFIGURATION.md](../../docs/PATH-CONFIGURATION.md) for complete guide.

---

## Implementation Status

- [ ] `dsmil_sandbox_runtime.c` - Planned
- [ ] `dsmil_provenance_runtime.c` - Planned
- [x] `dsmil_paths_runtime.c` - ✅ Complete (v1.6.1)
- [ ] `dsmil_crypto.c` - Planned
- [ ] `dsmil_elf.c` - Planned
- [ ] Sandbox profile loader - Planned
- [ ] CNSA 2.0 crypto integration - Planned

---

## Contributing

When implementing runtime libraries:

1. Follow secure coding practices (no buffer overflows, check all syscall returns)
2. Use constant-time crypto operations
3. Minimize dependencies (static linking preferred)
4. Add extensive error handling and logging
5. Write comprehensive unit tests

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for details.

---

## Security Considerations

### Sandbox Runtime

- Profile parsing must be robust against malformed input
- Seccomp filters must be installed before any privileged operations
- Capability drops are irreversible (design constraint)
- Resource limits prevent DoS attacks

### Provenance Runtime

- Signature verification must be constant-time
- Trust store must be immutable at runtime (read-only filesystem)
- Private keys must never be in memory longer than necessary
- Binary hash computation must cover all executable sections

---

## Performance

### Sandbox Setup Overhead

- Profile loading: ~1-2 ms
- Capability setup: ~1 ms
- Seccomp installation: ~2-5 ms
- Total: ~5-10 ms one-time startup cost

### Provenance Operations

**Build-Time**:
- Metadata collection: ~5 ms
- SHA-384 hashing (10 MB binary): ~8 ms
- ML-DSA-87 signing: ~12 ms
- ELF embedding: ~5 ms
- Total: ~30 ms per binary

**Runtime**:
- ELF extraction: ~1 ms
- SHA-384 verification: ~8 ms
- Certificate chain: ~15 ms (3-level)
- ML-DSA-87 verification: ~5 ms
- Total: ~30 ms one-time per exec

---

## Dependencies

Install required libraries:

```bash
# Ubuntu/Debian
sudo apt install libcap-ng-dev libseccomp-dev \
  libssl-dev libelf-dev libcbor-dev

# Build and install DSSSL (DSMIL-Grade OpenSSL with PQC support)
git clone https://github.com/SWORDIntel/DSSSL.git
cd DSSSL
./config --prefix=/usr/local/dsssl \
  --enable-dsmil-security \
  --enable-pqc \
  --enable-tpm2
make -j$(nproc)
sudo make install

# Build and install liboqs (for ML-DSA/ML-KEM - if not using DSSSL built-in PQC)
git clone https://github.com/open-quantum-safe/liboqs.git
cd liboqs
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
sudo make install
```

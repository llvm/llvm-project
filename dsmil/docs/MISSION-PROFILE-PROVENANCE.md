# Mission Profile Provenance Integration

**Version:** 1.3.0
**Feature:** Mission Profiles (Phase 1)
**SPDX-License-Identifier:** Apache-2.0 WITH LLVM-exception

## Overview

Mission profiles are first-class compile targets that define operational context and security constraints. All binaries compiled with a mission profile must embed complete provenance metadata to ensure auditability, traceability, and compliance verification.

## Provenance Requirements by Profile

### border_ops

**Classification:** RESTRICTED
**Provenance Required:** ✓ Mandatory
**Attestation Algorithm:** ML-DSA-87
**Key Source:** TPM hardware-backed key

**Mandatory Provenance Fields:**
- `mission_profile`: "border_ops"
- `mission_profile_hash`: SHA-384 hash of active mission-profiles.json
- `mission_classification`: "RESTRICTED"
- `mission_operational_context`: "hostile_environment"
- `mission_constraints_verified`: true
- `compile_timestamp`: ISO 8601 UTC timestamp
- `compiler_version`: DSLLVM version string
- `source_files`: List of all compiled source files with SHA-384 hashes
- `dependencies`: All linked libraries with SHA-384 hashes
- `clearance_floor`: "0xFF080000"
- `device_whitelist`: [0, 1, 2, 3, 30, 31, 32, 33, 47, 50, 53]
- `allowed_stages`: ["quantized", "serve"]
- `ct_enforcement`: "strict"
- `telemetry_level`: "minimal"
- `quantum_export`: false
- `max_deployment_days`: null (unlimited)

**Signature Requirements:**
- CNSA 2.0 compliant: ML-DSA-87 + SHA-384
- Hardware-backed signing key (TPM 2.0 or HSM)
- Include mission profile configuration hash in signed data
- Embed signature in ELF `.note.dsmil.provenance` section

### cyber_defence

**Classification:** CONFIDENTIAL
**Provenance Required:** ✓ Mandatory
**Attestation Algorithm:** ML-DSA-87
**Key Source:** TPM hardware-backed key

**Mandatory Provenance Fields:**
- `mission_profile`: "cyber_defence"
- `mission_profile_hash`: SHA-384 hash of active mission-profiles.json
- `mission_classification`: "CONFIDENTIAL"
- `mission_operational_context`: "defensive_operations"
- `mission_constraints_verified`: true
- `compile_timestamp`: ISO 8601 UTC timestamp
- `compiler_version`: DSLLVM version string
- `source_files`: List with SHA-384 hashes
- `dependencies`: All libraries with SHA-384 hashes
- `clearance_floor`: "0x07070000"
- `allowed_stages`: ["quantized", "serve", "finetune"]
- `ct_enforcement`: "strict"
- `telemetry_level`: "full"
- `quantum_export`: true
- `max_deployment_days`: 90
- `ai_config`: {"l5_performance_advisor": true, "l7_llm_assist": true, "l8_security_ai": true}

**Additional Requirements:**
- Expiration timestamp (compile_timestamp + 90 days)
- Runtime validation of expiration at process start
- Layer 8 Security AI scan results embedded in provenance

### exercise_only

**Classification:** UNCLASSIFIED
**Provenance Required:** ✓ Mandatory
**Attestation Algorithm:** ML-DSA-65 (relaxed)
**Key Source:** Software key (acceptable)

**Mandatory Provenance Fields:**
- `mission_profile`: "exercise_only"
- `mission_profile_hash`: SHA-384 hash of active mission-profiles.json
- `mission_classification`: "UNCLASSIFIED"
- `mission_operational_context`: "training_simulation"
- `mission_constraints_verified`: true
- `compile_timestamp`: ISO 8601 UTC timestamp
- `compiler_version`: DSLLVM version string
- `max_deployment_days`: 30
- `simulation_mode`: true
- `allowed_stages`: ["quantized", "serve", "finetune", "debug"]

**Expiration:**
- Hard expiration: 30 days from compile_timestamp
- Runtime check fails on expired binaries

### lab_research

**Classification:** UNCLASSIFIED
**Provenance Required:** ✗ Optional
**Attestation Algorithm:** None (optional ML-DSA-65)
**Key Source:** N/A

**Optional Provenance Fields:**
- `mission_profile`: "lab_research"
- `compile_timestamp`: ISO 8601 UTC timestamp
- `compiler_version`: DSLLVM version string
- `experimental_features`: ["rl_loop", "quantum_offload", "custom_passes"]

**Notes:**
- No signature required
- No expiration enforcement
- Debug symbols retained
- No production deployment allowed

## Provenance Embedding Format

### ELF Section: `.note.dsmil.provenance`

```c
struct DsmilProvenanceNote {
    Elf64_Nhdr nhdr;                      // Standard ELF note header
    char name[12];                        // "DSMIL-1.3\0"
    uint32_t version;                     // 0x00010300 (v1.3)
    uint32_t json_size;                   // Size of JSON payload
    uint8_t json_data[json_size];         // JSON provenance record
    uint32_t signature_algorithm;         // 0x0001 = ML-DSA-87, 0x0002 = ML-DSA-65
    uint32_t signature_size;              // Size of signature
    uint8_t signature[signature_size];    // ML-DSA signature
};
```

### JSON Provenance Schema (v1.3)

```json
{
  "$schema": "https://dsmil.org/schemas/provenance-v1.3.json",
  "version": "1.3.0",
  "mission_profile": {
    "profile_id": "border_ops",
    "profile_hash": "sha384:a1b2c3...",
    "classification": "RESTRICTED",
    "operational_context": "hostile_environment",
    "constraints_verified": true
  },
  "build": {
    "compiler": "DSLLVM 1.3.0-dev",
    "compiler_hash": "sha384:d4e5f6...",
    "timestamp": "2026-01-15T14:30:00Z",
    "host": "build-server-01.local",
    "user": "ci-bot"
  },
  "sources": [
    {
      "path": "src/main.c",
      "hash": "sha384:1a2b3c...",
      "layer": 7,
      "device": 47
    }
  ],
  "dependencies": [
    {
      "name": "libdsmil_runtime.so",
      "version": "1.3.0",
      "hash": "sha384:4d5e6f..."
    }
  ],
  "security": {
    "clearance_floor": "0xFF080000",
    "device_whitelist": [0, 1, 2, 3, 30, 31, 32, 33, 47, 50, 53],
    "allowed_stages": ["quantized", "serve"],
    "ct_enforcement": "strict",
    "telemetry_level": "minimal",
    "quantum_export": false
  },
  "deployment": {
    "max_deployment_days": null,
    "expiration_timestamp": null
  },
  "attestation": {
    "algorithm": "ML-DSA-87",
    "key_id": "tpm:sha256:7g8h9i...",
    "signature_offset": 2048,
    "signature_size": 4627
  },
  "cnsa2_compliance": {
    "hash_algorithm": "SHA-384",
    "signature_algorithm": "ML-DSA-87",
    "key_encapsulation": "ML-KEM-1024",
    "compliant": true
  }
}
```

## Runtime Validation

### Binary Load-Time Checks

When a DSMIL binary is loaded, the runtime performs:

1. **Provenance Extraction**
   - Locate `.note.dsmil.provenance` section
   - Parse provenance JSON
   - Validate schema version compatibility

2. **Signature Verification**
   - Extract ML-DSA signature
   - Verify signature over (JSON + mission_profile_hash)
   - Check key trust chain (TPM/HSM root)

3. **Mission Profile Validation**
   - Load current mission-profiles.json
   - Compute SHA-384 hash
   - Compare with `mission_profile_hash` in provenance
   - If mismatch: REJECT LOAD (prevents running binaries compiled with stale profiles)

4. **Expiration Check**
   - If `max_deployment_days` is set, compute `compile_timestamp + max_deployment_days`
   - Compare with current time
   - If expired: REJECT LOAD

5. **Clearance Check**
   - Compare process effective clearance with `clearance_floor`
   - If process clearance < clearance_floor: REJECT LOAD

6. **Device Availability**
   - If `device_whitelist` is set, check all required devices are accessible
   - If any device unavailable: REJECT LOAD (unless `DSMIL_ALLOW_DEGRADED=1`)

### Example: border_ops Binary Load

```
[DSMIL Runtime] Loading binary: /opt/llm_worker/bin/inference_server
[DSMIL Runtime] Provenance found: v1.3.0
[DSMIL Runtime] Mission Profile: border_ops (RESTRICTED)
[DSMIL Runtime] Verifying ML-DSA-87 signature...
[DSMIL Runtime]   Key ID: tpm:sha256:7g8h9i...
[DSMIL Runtime]   Signature valid ✓
[DSMIL Runtime] Mission profile hash: sha384:a1b2c3...
[DSMIL Runtime]   Current config hash: sha384:a1b2c3... ✓
[DSMIL Runtime] Clearance check: 0xFF080000 <= 0xFF080000 ✓
[DSMIL Runtime] Device whitelist: [0,1,2,3,30,31,32,33,47,50,53]
[DSMIL Runtime]   All devices available ✓
[DSMIL Runtime] Expiration: none (indefinite deployment) ✓
[DSMIL Runtime] ✓ All provenance checks passed
[DSMIL Runtime] Starting process with mission profile: border_ops
```

### Example: cyber_defence Binary Expiration

```
[DSMIL Runtime] Loading binary: /opt/defense/bin/threat_analyzer
[DSMIL Runtime] Provenance found: v1.3.0
[DSMIL Runtime] Mission Profile: cyber_defence (CONFIDENTIAL)
[DSMIL Runtime] Verifying ML-DSA-87 signature...
[DSMIL Runtime]   Signature valid ✓
[DSMIL Runtime] Expiration check:
[DSMIL Runtime]   Compiled: 2025-10-01T00:00:00Z
[DSMIL Runtime]   Max deployment: 90 days
[DSMIL Runtime]   Expiration: 2025-12-30T00:00:00Z
[DSMIL Runtime]   Current time: 2026-01-05T10:00:00Z
[DSMIL Runtime]   ✗ BINARY EXPIRED (6 days overdue)
[DSMIL Runtime] FATAL: Cannot execute expired cyber_defence binary
[DSMIL Runtime] Hint: Recompile with current DSLLVM toolchain
```

## Compile-Time Provenance Generation

### DsmilProvenancePass Integration

The `DsmilProvenancePass.cpp` (link-time) is extended to:

1. **Read Mission Profile Metadata**
   - Extract `dsmil.mission_profile` module flag set by `DsmilMissionPolicyPass`
   - Load mission-profiles.json
   - Compute SHA-384 hash of mission-profiles.json

2. **Build Provenance JSON**
   - Include all mission profile constraints
   - Add compile timestamp
   - List all source files with SHA-384 hashes
   - List all dependencies

3. **Sign Provenance**
   - If `provenance_required: true` in mission profile:
     - Load signing key from TPM/HSM (or software key for lab_research)
     - Compute ML-DSA-87 signature over (JSON + mission_profile_hash)
     - Embed signature in provenance note

4. **Embed in Binary**
   - Create `.note.dsmil.provenance` ELF section
   - Write provenance note structure
   - Set section flags: SHF_ALLOC (loaded at runtime)

### Example Compilation

```bash
# Compile with border_ops mission profile
dsmil-clang \
  -fdsmil-mission-profile=border_ops \
  -fdsmil-mission-profile-config=/etc/dsmil/mission-profiles.json \
  -fdsmil-provenance=full \
  -fdsmil-provenance-sign-key=tpm://0 \
  src/llm_worker.c \
  -o bin/llm_worker

# Output:
# [DSMIL Mission Policy] Enforcing mission profile: border_ops (Border Operations)
#   Classification: RESTRICTED
#   CT Enforcement: strict
#   Telemetry Level: minimal
# [DSMIL Provenance] Generating provenance record
#   Mission Profile Hash: sha384:a1b2c3...
#   Signing with ML-DSA-87 (TPM key)
# [DSMIL Provenance] ✓ Provenance embedded in .note.dsmil.provenance
```

## Forensics and Audit

### Extracting Provenance from Binary

```bash
# Extract provenance JSON
readelf -x .note.dsmil.provenance bin/llm_worker > provenance.hex
xxd -r provenance.hex | jq .

# Verify signature
dsmil-verify --binary bin/llm_worker --tpm-key tpm://0

# Check mission profile
dsmil-inspect bin/llm_worker
# Output:
#   Mission Profile: border_ops
#   Classification: RESTRICTED
#   Compiled: 2026-01-15T14:30:00Z
#   Signature: VALID (ML-DSA-87, TPM key)
#   Expiration: None
#   Status: DEPLOYABLE
```

### Layer 62 Forensics Integration

Mission profile provenance integrates with Layer 62 (Forensics/Evidence) for post-incident analysis:

- All provenance records are indexed by binary hash
- Mission profile violations trigger forensic logging
- Expired binaries are flagged in forensic timeline
- Provenance signatures enable non-repudiation

## Migration from v1.2 to v1.3

### Backward Compatibility

- Binaries compiled with DSLLVM 1.2 (no mission profile) continue to work
- v1.3 runtime detects missing mission profile provenance
- If missing, assumes `lab_research` profile (permissive mode)

### Upgrade Path

1. Deploy mission-profiles.json to `/etc/dsmil/mission-profiles.json`
2. Recompile all production binaries with `-fdsmil-mission-profile=<profile>`
3. Configure runtime to reject binaries without mission profile provenance
4. Audit all deployed binaries for mission profile compliance

## References

- **Mission Profiles Configuration:** `/etc/dsmil/mission-profiles.json`
- **CNSA 2.0 Spec:** CNSSP-15 (NSA)
- **ML-DSA Spec:** FIPS 204
- **Provenance Pass:** `dsmil/lib/Passes/DsmilProvenancePass.cpp`
- **Mission Policy Pass:** `dsmil/lib/Passes/DsmilMissionPolicyPass.cpp`
- **DSLLVM Roadmap:** `dsmil/docs/DSLLVM-ROADMAP.md`

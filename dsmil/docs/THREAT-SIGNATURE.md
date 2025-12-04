# DSLLVM Threat Signature Embedding Guide (Feature 2.2)

**Version**: 1.4
**Feature**: Threat Signature Embedding for Future Forensics
**Status**: Implemented
**Date**: 2025-11-25

---

## Overview

Threat Signature Embedding enables **future AI-driven forensics** by embedding non-identifying fingerprints in binaries. Layer 62 (Forensics/SIEM) uses these signatures to correlate observed malware with known-good templates, enabling:

- **Imposter Detection**: Spot tampered versions of own binaries
- **Supply Chain Security**: Detect unauthorized modifications
- **Post-Incident Analysis**: "Is this suspicious binary ours?"

---

## Motivation

**Problem**: After a security incident, forensics teams find suspicious binaries but struggle to determine if they're tampered versions of legitimate software.

**Solution**: Embed cryptographic fingerprints during compilation that Layer 62 can use for correlation:
- Control-flow structure (CFG hash)
- Crypto usage patterns
- Protocol schemas

**Key Insight**: Non-identifying fingerprints (hashes, not raw structures) enable correlation without leaking implementation details.

---

## Architecture

```
┌──────────────────────────────────────┐
│ Compile Time                         │
│ ┌──────────────────────────────────┐ │
│ │ DsmilThreatSignaturePass         │ │
│ │ ├─ Extract CFG structure         │ │
│ │ ├─ Hash with SHA-256             │ │
│ │ ├─ Identify crypto patterns      │ │
│ │ └─ Identify protocol schemas     │ │
│ └────────────┬─────────────────────┘ │
│              │                        │
│              ▼                        │
│ ┌──────────────────────────────────┐ │
│ │ threat-signature.json            │ │
│ │ {                                │ │
│ │   "cfg_hash": "0x1a2b3c...",    │ │
│ │   "crypto": ["ML-KEM", "AES"],  │ │
│ │   "protocols": ["TLS-1.3"]      │ │
│ │ }                                │ │
│ └────────────┬─────────────────────┘ │
└──────────────┼───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ Secure Storage (SIEM)                │
│ ├─ Encrypt with ML-KEM-1024          │
│ ├─ Store in Layer 62 database        │
│ └─ Index by binary hash              │
└──────────────┬───────────────────────┘
               │
      (Months later...)
               │
               ▼
┌──────────────────────────────────────┐
│ Forensics Analysis                   │
│ ┌──────────────────────────────────┐ │
│ │ Suspicious binary found          │ │
│ │ ├─ Extract CFG hash              │ │
│ │ ├─ Query Layer 62 SIEM           │ │
│ │ └─ Match: "sensor.bin tampered!" │ │
│ └──────────────────────────────────┘ │
└──────────────────────────────────────┘
```

---

## Threat Signature Components

### 1. Control-Flow Fingerprint

**What**: SHA-256 hash of CFG structure
**Why**: Unique per binary, changes if code is modified
**How**: Concatenate function names + basic block counts + CFG edges

```json
{
  "control_flow_fingerprint": {
    "algorithm": "CFG-SHA256",
    "hash": "a1b2c3d4e5f6...",
    "num_functions": 127,
    "functions_included": ["main", "crypto_init", "network_send"]
  }
}
```

### 2. Crypto Patterns

**What**: List of cryptographic algorithms used
**Why**: Helps identify if crypto implementation was tampered
**How**: Scan function names and attributes for crypto indicators

```json
{
  "crypto_patterns": [
    {
      "algorithm": "ML-KEM-1024"
    },
    {
      "algorithm": "ML-DSA-87"
    },
    {
      "algorithm": "AES-256-GCM"
    },
    {
      "algorithm": "constant_time_enforced"
    }
  ]
}
```

### 3. Protocol Schemas

**What**: Network protocols and serialization formats
**Why**: Detect if protocol implementation was modified
**How**: Identify protocol usage from function names

```json
{
  "protocol_schemas": [
    {
      "protocol": "TLS-1.3"
    },
    {
      "protocol": "HTTP/2"
    }
  ]
}
```

---

## Usage

### Enable Threat Signatures

```bash
# Compile with threat signature embedding
dsmil-clang -dsmil-threat-signature \
            -dsmil-threat-signature-output=sensor.sig.json \
            -O3 -o sensor.bin sensor.c
```

### Generated Signature

**File**: `sensor.sig.json`

```json
{
  "version": 1,
  "schema": "dsmil-threat-signature-v1",
  "module": "sensor.bin",
  "control_flow_fingerprint": {
    "algorithm": "CFG-SHA256",
    "hash": "f4a3b9c2d1e8f7...",
    "num_functions": 42,
    "functions_included": [
      "main",
      "sensor_init",
      "collect_data",
      "encrypt_data",
      "transmit_data"
    ]
  },
  "crypto_patterns": [
    {"algorithm": "AES-256-GCM"},
    {"algorithm": "ML-KEM-1024"},
    {"algorithm": "SHA-384"},
    {"algorithm": "constant_time_enforced"}
  ],
  "protocol_schemas": [
    {"protocol": "TLS"},
    {"protocol": "HTTP"}
  ]
}
```

### Store in SIEM

```bash
# Encrypt signature
ml-kem-encrypt --key=siem_pubkey sensor.sig.json > sensor.sig.enc

# Upload to Layer 62 SIEM
siem-upload --layer=62 --type=threat_signature sensor.sig.enc
```

---

## Forensics Workflow

### 1. Incident Detection

Suspicious binary found on network:
```bash
${DSMIL_TMP_DIR:-/tmp}/suspicious_binary
```

### 2. Extract Signature

```bash
# Extract threat signature from suspicious binary (uses dynamic path resolution)
dsmil-extract-signature ${DSMIL_TMP_DIR:-/tmp}/suspicious_binary > suspicious.sig.json
# Or use runtime API:
# #include <dsmil_paths.h>
# char tmp_path[PATH_MAX];
# snprintf(tmp_path, sizeof(tmp_path), "%s/suspicious_binary", dsmil_get_tmp_dir());
```

### 3. Query SIEM

```bash
# Query Layer 62 for matching signatures
siem-query --layer=62 --type=threat_signature \
           --cfg-hash=$(jq -r '.control_flow_fingerprint.hash' suspicious.sig.json)
```

### 4. Correlation Result

```json
{
  "match_found": true,
  "original_binary": "sensor.bin",
  "similarity_score": 0.95,
  "differences": [
    "Function 'validate_input' removed",
    "Crypto pattern 'constant_time_enforced' missing"
  ],
  "verdict": "TAMPERED",
  "confidence": 0.97
}
```

### 5. Response

```
ALERT: Tampered binary detected!
- Original: sensor.bin (v1.2.3)
- Found: /tmp/suspicious_binary
- Tampering: Input validation removed
- Action: Quarantine system, investigate lateral movement
```

---

## Security Considerations

### Non-Identifying Fingerprints

**Risk**: Signatures could leak internal structure
**Mitigation**: Only store hashes, not raw CFGs

```
❌ Don't store: Raw control-flow graph
✅ Store: SHA-256 hash of CFG
```

### Secure Storage

**Risk**: Signatures could be stolen from SIEM
**Mitigation**: Encrypt with ML-KEM-1024

```bash
# Encrypt before storage
ml-kem-encrypt --key=siem_pubkey signature.json > signature.enc
```

### False Positives

**Risk**: Legitimate binaries flagged as tampered
**Mitigation**: Multiple features + human review

```
Correlation requires:
- CFG hash match (>90%)
- Crypto patterns match
- Protocol schemas match
- Human analyst review
```

### Storage Overhead

**Impact**: ~5-10 KB per binary
**Mitigation**: Optional feature, enable for high-value targets only

---

## Integration with CI/CD

```yaml
# .github/workflows/threat-signature.yml
jobs:
  build-with-signatures:
    runs-on: meteor-lake
    steps:
      - name: Build Binary
        run: |
          dsmil-clang -dsmil-threat-signature \
                      -dsmil-threat-signature-output=sensor.sig.json \
                      -O3 -o sensor.bin sensor.c

      - name: Encrypt Signature
        run: |
          ml-kem-encrypt --key=${{ secrets.SIEM_PUBKEY }} \
                        sensor.sig.json > sensor.sig.enc

      - name: Upload to SIEM
        run: |
          siem-upload --layer=62 \
                      --type=threat_signature \
                      --binary=sensor.bin \
                      --signature=sensor.sig.enc

      - name: Deploy Binary
        run: |
          deploy-to-production sensor.bin
```

---

## Use Cases

### Use Case 1: Supply Chain Attack Detection

**Scenario**: Vendor provides "updated" binary
**Question**: Is this legitimately our code or tampered?

**Solution**:
```bash
# Extract signature from vendor binary
dsmil-extract-signature vendor_binary.bin > vendor.sig.json

# Compare with known-good signature
siem-query --compare vendor.sig.json official_v1.2.3.sig.json

# Result: "82% match - functions added, investigate"
```

### Use Case 2: Post-Breach Forensics

**Scenario**: Breach detected, multiple binaries on systems
**Question**: Which binaries are ours? Which are attacker implants?

**Solution**:
```bash
# Scan all binaries
for bin in ${DSMIL_BIN_DIR:-/opt/dsmil/bin}/*; do
    dsmil-extract-signature $bin | \
    siem-query --layer=62 --match
done

# Result:
# - sensor.bin: MATCH (legitimate)
# - logger.bin: NO MATCH (attacker implant!)
# - network_daemon.bin: PARTIAL MATCH (tampered, 73% similar)
```

### Use Case 3: Malware Attribution

**Scenario**: Malware found using our crypto libraries
**Question**: Did attacker steal our code?

**Solution**:
```bash
# Extract crypto patterns from malware
dsmil-extract-signature malware.bin > malware.sig.json

# Check crypto patterns
jq '.crypto_patterns' malware.sig.json

# Result: Matches our ML-KEM implementation
# Conclusion: Likely stolen/reused our crypto code
```

---

## Best Practices

### 1. Enable for High-Value Binaries

```bash
# Production deployments
dsmil-clang -dsmil-threat-signature ...

# Internal tools (optional)
dsmil-clang ...
```

### 2. Store Signatures Securely

```bash
# Always encrypt
ml-kem-encrypt signature.json > signature.enc

# Restrict access
chmod 600 signature.enc
chown siem:siem signature.enc
```

### 3. Version Signatures

```bash
# Include version in signature
dsmil-clang -dsmil-threat-signature \
            -DBINARY_VERSION="1.2.3" \
            -o sensor.bin sensor.c

# Store with version metadata
siem-upload --version=1.2.3 signature.enc
```

### 4. Periodic Validation

```bash
# Weekly: Re-extract signatures from production
cron-job: extract-and-validate-signatures

# Compare with stored signatures
# Alert on mismatches
```

### 5. Human Review Required

```
Automated correlation provides:
- Similarity score
- Identified differences
- Confidence level

BUT: Always require human analyst review before action
```

---

## CLI Reference

```bash
# Enable threat signatures
-dsmil-threat-signature

# Output path
-dsmil-threat-signature-output=<path>

# Example
dsmil-clang -dsmil-threat-signature \
            -dsmil-threat-signature-output=output.json \
            -O3 -o binary source.c
```

---

## Summary

**Threat Signature Embedding** enables future forensics by embedding non-identifying fingerprints:

- **CFG Hash**: SHA-256 of control-flow structure
- **Crypto Patterns**: Algorithms and enforcement metadata
- **Protocol Schemas**: Network protocols used

**Benefits**:
- Detect tampered binaries
- Supply chain security
- Post-incident forensics
- Malware attribution

**Security**:
- Non-identifying (hashes only)
- Encrypted storage (ML-KEM-1024)
- Multiple features prevent false positives
- Human review required

---

**Document Version**: 1.0
**Date**: 2025-11-25
**Next Review**: After first forensics case

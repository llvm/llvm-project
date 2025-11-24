# Mission Profiles - Test Examples

This directory contains example programs demonstrating DSLLVM mission profiles.

## Examples

### border_ops_example.c

LLM inference worker for border operations deployment.

**Profile:** `border_ops`
**Classification:** RESTRICTED
**Features:**
- Air-gapped deployment
- Minimal telemetry
- Strict constant-time enforcement
- Device whitelist enforcement
- No expiration

**Compile:**
```bash
dsmil-clang -fdsmil-mission-profile=border_ops \
  -fdsmil-provenance=full -O3 border_ops_example.c \
  -o border_ops_worker
```

### cyber_defence_example.c

Threat analyzer for cyber defence operations.

**Profile:** `cyber_defence`
**Classification:** CONFIDENTIAL
**Features:**
- Network-connected deployment
- Full telemetry
- Layer 8 Security AI integration
- Quantum optimization support
- 90-day expiration

**Compile:**
```bash
dsmil-clang -fdsmil-mission-profile=cyber_defence \
  -fdsmil-l8-security-ai=enabled -fdsmil-provenance=full \
  -O3 cyber_defence_example.c -o threat_analyzer
```

## Building All Examples

```bash
# Build all examples
make -C dsmil/test/mission-profiles

# Build specific profile
make border_ops
make cyber_defence
```

## Testing

```bash
# Run examples
./border_ops_worker
./threat_analyzer

# Inspect provenance
dsmil-inspect border_ops_worker
dsmil-inspect threat_analyzer
```

## Documentation

See:
- `dsmil/docs/MISSION-PROFILES-GUIDE.md` - Complete user guide
- `dsmil/docs/MISSION-PROFILE-PROVENANCE.md` - Provenance integration
- `dsmil/config/mission-profiles.json` - Configuration schema

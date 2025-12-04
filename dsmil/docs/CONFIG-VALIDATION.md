# Configuration Validation Guide

**DSLLVM v1.7+ Configuration Validation**

## Overview

The `dsmil-config-validate` tool provides comprehensive validation of DSLLVM configuration components, catching errors before deployment and ensuring operational readiness.

---

## Quick Start

```bash
# Validate all configuration
dsmil-config-validate --all

# Validate specific components
dsmil-config-validate --mission-profiles
dsmil-config-validate --truststore
dsmil-config-validate --paths

# Generate health report
dsmil-config-validate --report=health.json

# Auto-fix common issues
dsmil-config-validate --auto-fix
```

---

## Validation Components

### Mission Profiles

Validates mission profile JSON files:
- JSON syntax correctness
- Schema compliance
- Profile name validity
- Setting consistency

```bash
dsmil-config-validate --mission-profiles
```

### Path Configuration

Checks that all configured paths exist and are accessible:
- Configuration directory
- Binary directory
- Truststore directory
- Log directory
- Runtime directory

```bash
dsmil-config-validate --paths
```

### Truststore

Validates truststore configuration:
- Certificate file presence
- Certificate chain validity
- Revocation list integrity
- Key accessibility

```bash
dsmil-config-validate --truststore
```

### Classification

Validates classification configuration:
- Cross-domain gateway consistency
- Classification level hierarchy
- Gateway approval status

```bash
dsmil-config-validate --classification
```

---

## Health Reports

Generate comprehensive health reports in JSON format:

```bash
dsmil-config-validate --all --report=health.json
```

Report includes:
- Validation status for each component
- Error messages and codes
- Recommendations for fixes
- Timestamp and version information

---

## Auto-Fix

Automatically fix common configuration issues:

```bash
dsmil-config-validate --auto-fix
```

Fixes include:
- Creating missing directories
- Setting correct permissions
- Generating default configurations

---

## CI/CD Integration

Integrate validation into CI/CD pipelines:

```yaml
# GitLab CI example
validate_config:
  stage: validate
  script:
    - dsmil-config-validate --all --report=validation.json
    - |
      if [ $? -ne 0 ]; then
        echo "Configuration validation failed"
        exit 1
      fi
  artifacts:
    reports:
      json: validation.json
```

---

## Related Documentation

- **[PATH-CONFIGURATION.md](PATH-CONFIGURATION.md)**: Path configuration guide
- **[MISSION-PROFILES-GUIDE.md](MISSION-PROFILES-GUIDE.md)**: Mission profile setup

---

**DSLLVM Configuration Validation**: Ensuring operational readiness through automated validation.

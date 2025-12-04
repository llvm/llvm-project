# Setup Wizard Guide

**DSLLVM v1.7+ Interactive Setup**

## Overview

The `dsmil-setup` wizard guides users through DSLLVM installation and configuration, reducing setup time and configuration errors.

---

## Quick Start

```bash
# Interactive wizard
dsmil-setup

# Non-interactive mode
dsmil-setup --non-interactive --profile=cyber_defence

# Verify installation
dsmil-setup --verify

# Fix issues
dsmil-setup --fix
```

---

## Interactive Wizard

The interactive wizard guides you through:

1. **Installation Detection**
   - Detects existing DSLLVM installation
   - Checks dependencies
   - Verifies permissions

2. **Mission Profile Setup**
   - Select mission profile (border_ops, cyber_defence, etc.)
   - Customize settings
   - Generate profile JSON

3. **Path Configuration**
   - Detect installation prefix
   - Configure custom paths
   - Test path resolution

4. **Verification**
   - Validate configuration
   - Check all components
   - Report issues

---

## Non-Interactive Mode

For CI/CD and automated setups:

```bash
dsmil-setup --non-interactive \
  --profile=cyber_defence \
  --prefix=/opt/dsmil \
  --output=/etc/dsmil/config.json
```

---

## Verification

Verify existing installation:

```bash
dsmil-setup --verify
```

Checks:
- Path configuration
- Mission profiles
- Truststore
- Classification settings

---

## Auto-Fix

Automatically fix common issues:

```bash
dsmil-setup --fix
```

Fixes:
- Missing directories
- Incorrect permissions
- Configuration file issues

---

## Templates

Generate configuration from templates:

```bash
dsmil-setup --template=border_ops --output=/etc/dsmil/mission-profiles.json
```

Available templates:
- `border_ops`: Maximum security
- `cyber_defence`: AI-enhanced
- `exercise_only`: Training exercises
- `lab_research`: Experimental

---

## Related Documentation

- **[PATH-CONFIGURATION.md](PATH-CONFIGURATION.md)**: Path configuration guide
- **[MISSION-PROFILES-GUIDE.md](MISSION-PROFILES-GUIDE.md)**: Mission profile setup
- **[CONFIG-VALIDATION.md](CONFIG-VALIDATION.md)**: Configuration validation

---

**DSLLVM Setup Wizard**: Streamlined installation and configuration.

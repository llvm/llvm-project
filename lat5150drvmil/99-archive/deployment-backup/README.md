# Dell MIL-SPEC Platform - Deployment and Migration Tools

**Version:** 1.0.0
**Author:** Claude Agent Framework - DEPLOYER Agent
**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY

## Overview

This directory contains comprehensive migration scripts for transitioning Dell Latitude 5450 MIL-SPEC systems from manual installation to .deb package-based deployment.

## Migration Scripts

### 1. migrate-to-packages.sh

**Main migration orchestrator** - Handles complete transition from manual installation to .deb packages.

**Features:**
- Automatic detection of manual installation
- Comprehensive backup creation with timestamp
- Safe removal of manual components
- Package installation and configuration
- Configuration migration and preservation
- Full validation and reporting
- Dry-run support for testing

**Usage:**
```bash
# Preview migration (recommended first)
sudo ./migrate-to-packages.sh --dry-run --verbose

# Interactive migration
sudo ./migrate-to-packages.sh

# Automatic migration (no prompts)
sudo ./migrate-to-packages.sh --yes --verbose
```

**Options:**
- `-h, --help` - Show help message
- `-n, --dry-run` - Preview without making changes
- `-f, --force` - Force migration bypassing checks
- `-y, --yes` - Auto-confirm all prompts
- `--skip-backup` - Skip backup creation (NOT RECOMMENDED)
- `-v, --verbose` - Detailed output

**What It Does:**

1. **Detection Phase**
   - Scans for manual installation artifacts
   - Identifies modules, configs, services
   - Validates system state

2. **Backup Phase**
   - Creates `/var/backups/dell-milspec-manual-TIMESTAMP/`
   - Backs up kernel modules (dsmil-72dev.ko, tpm2_accel_early.ko)
   - Backs up `/opt/dsmil/` monitoring system
   - Backs up configuration files
   - Backs up systemd services
   - Creates compressed archive
   - Generates detailed manifest

3. **Removal Phase**
   - Stops running services gracefully
   - Unloads manual kernel modules
   - Removes `/opt/dsmil/` directory
   - Removes manual configuration files
   - Removes systemd service files
   - Updates module dependencies

4. **Installation Phase**
   - Updates apt cache
   - Installs `dell-milspec-tools`
   - Installs `dell-milspec-dsmil-dkms` (if available)
   - Installs `tpm2-accel-early-dkms` (if available)
   - Configures DKMS modules

5. **Migration Phase**
   - Creates `/etc/dell-milspec/`
   - Migrates configurations from `/opt/dsmil/config/` to `/etc/dell-milspec/`
   - Converts JSON configs to shell format where needed
   - Preserves user customizations
   - Updates modprobe.d configurations

6. **Validation Phase**
   - Verifies package installation
   - Tests command availability
   - Checks DKMS module registration
   - Validates configuration directory
   - Confirms backup preservation

7. **Reporting Phase**
   - Generates comprehensive migration report
   - Documents all changes
   - Provides next steps
   - Creates rollback instructions

**Backup Structure:**
```
/var/backups/dell-milspec-manual-TIMESTAMP/
├── MANIFEST.txt                  # Complete inventory
├── MIGRATION_REPORT.txt          # Migration results
├── modules/
│   ├── dsmil-72dev.ko           # Manual DSMIL module
│   └── tpm2_accel_early.ko      # Manual TPM2 module
├── monitoring/
│   └── dsmil/                   # Complete /opt/dsmil/ copy
│       ├── monitoring/          # Python monitoring scripts
│       ├── config/              # JSON configurations
│       ├── logs/                # Historical logs
│       └── bin/                 # Scripts and tools
├── configs/
│   ├── dsmil-72dev.conf         # Modprobe config
│   ├── tpm2-acceleration.conf   # Modprobe config
│   └── *.conf                   # Other configs
├── services/
│   └── dsmil-monitor.service    # Systemd service
└── logs/
    └── *.log                    # Any log files
```

**Configuration Mapping:**

| Old Location (Manual) | New Location (Package) |
|----------------------|------------------------|
| `/opt/dsmil/config/dsmil.json` | `/etc/dell-milspec/dsmil.conf` |
| `/opt/dsmil/config/monitoring.json` | `/etc/dell-milspec/monitoring.json` |
| `/opt/dsmil/config/safety.json` | `/etc/dell-milspec/safety.json` |
| `/etc/modprobe.d/dsmil-72dev.conf` | `/etc/modprobe.d/dell-milspec.conf` |
| `/opt/dsmil/monitoring/*.py` | `/usr/share/dell-milspec/monitoring/*.py` |

**Exit Codes:**
- `0` - Migration successful
- `1` - Migration failed (check logs)

**Logs:**
- `/var/log/dell-milspec-migration.log` - Detailed execution log

---

### 2. detect-manual-install.sh

**Detection utility** - Identifies manual installation artifacts and generates structured report.

**Features:**
- Comprehensive artifact detection
- Multiple output formats (JSON, text, summary)
- Exit codes for scripting
- Verbose mode for debugging

**Usage:**
```bash
# JSON output (default)
./detect-manual-install.sh

# Human-readable text
./detect-manual-install.sh --format text

# One-line summary
./detect-manual-install.sh --format summary

# Verbose detection
./detect-manual-install.sh --verbose
```

**Options:**
- `-h, --help` - Show help message
- `-f, --format FORMAT` - Output format: json, text, summary
- `-v, --verbose` - Verbose output to stderr

**Detection Areas:**

1. **Kernel Modules**
   - `/lib/modules/*/extra/dsmil-72dev.ko`
   - `/lib/modules/*/kernel/drivers/tpm/tpm2_accel_early.ko`
   - Loaded modules (lsmod check)

2. **Installation Directory**
   - `/opt/dsmil/`
   - Subdirectories: monitoring, config, logs, bin
   - File counts and sizes

3. **Systemd Services**
   - `/etc/systemd/system/dsmil-monitor.service`
   - Service status (active/enabled)

4. **Configuration Files**
   - `/etc/modprobe.d/dsmil-72dev.conf`
   - `/etc/modules-load.d/tpm2-acceleration.conf`
   - `/etc/modprobe.d/tpm2-acceleration.conf`
   - `/etc/udev/rules.d/*dsmil*`

5. **Manual Tools**
   - `/usr/local/bin/dsmil-*`
   - `/usr/local/bin/milspec-*`

6. **Device Files**
   - `/dev/dsmil*`
   - `/dev/tpm2_accel_early`

7. **System Groups**
   - `dsmil` group
   - Group membership

**JSON Output Format:**
```json
{
  "detection_version": "1.0.0",
  "timestamp": "2025-10-11T15:30:00-04:00",
  "hostname": "dell-lat5450",
  "kernel": "6.16.9-amd64",
  "status": "manual_found",
  "exit_code": 0,
  "summary": {
    "total_artifacts": 15,
    "modules": 2,
    "configs": 3,
    "services": 1,
    "directories": 5,
    "tools": 4
  },
  "artifacts": [
    {"type": "module", "name": "dsmil-72dev", "path": "/lib/modules/.../dsmil-72dev.ko"},
    {"type": "directory", "name": "install", "path": "/opt/dsmil"}
  ],
  "modules": [...],
  "configurations": [...],
  "services": [...],
  "directories": [...],
  "tools": [...]
}
```

**Exit Codes:**
- `0` - Manual installation found
- `1` - Clean system (no manual installation)
- `2` - Partial installation detected

**Use Cases:**
- Pre-migration validation
- Inventory management
- Scripted detection
- Migration planning

---

### 3. rollback-migration.sh

**Emergency rollback** - Reverts system from .deb packages back to manual installation.

**Features:**
- Automatic backup location
- Safe package removal
- Complete restoration from backup
- Module reloading
- Configuration restoration
- Validation and reporting

**Usage:**
```bash
# Preview rollback (recommended first)
sudo ./rollback-migration.sh --dry-run --verbose

# Interactive rollback
sudo ./rollback-migration.sh

# Specify backup location
sudo ./rollback-migration.sh --backup /var/backups/dell-milspec-manual-20251011_143022

# Automatic rollback (no prompts)
sudo ./rollback-migration.sh --yes --verbose
```

**Options:**
- `-h, --help` - Show help message
- `-n, --dry-run` - Preview without making changes
- `-f, --force` - Force rollback bypassing checks
- `-y, --yes` - Auto-confirm all prompts
- `-b, --backup PATH` - Specify backup directory
- `-v, --verbose` - Detailed output

**What It Does:**

1. **Detection Phase**
   - Scans for installed .deb packages
   - Identifies dell-milspec-tools, dsmil-dkms, tpm2-dkms
   - Validates system state

2. **Backup Location Phase**
   - Searches `/var/backups/dell-milspec-manual-*`
   - Validates backup integrity
   - Checks MANIFEST.txt
   - Allows manual selection if multiple backups

3. **Package Removal Phase**
   - Stops milspec-monitor processes
   - Unloads DKMS modules
   - Purges dell-milspec-tools
   - Purges dell-milspec-dsmil-dkms
   - Purges tpm2-accel-early-dkms
   - Cleans up package cache
   - Removes package configurations

4. **Restoration Phase**
   - Restores kernel modules to `/lib/modules/*/extra/`
   - Restores monitoring system to `/opt/dsmil/`
   - Restores configurations to `/etc/modprobe.d/`, `/etc/modules-load.d/`
   - Restores systemd service to `/etc/systemd/system/`
   - Sets appropriate permissions
   - Updates module dependencies

5. **Validation Phase**
   - Verifies kernel modules exist
   - Confirms monitoring directory restored
   - Checks no packages remain
   - Validates key subdirectories

6. **Module Loading Phase**
   - Loads dsmil-72dev module
   - Loads tpm2_accel_early module
   - Verifies modules in lsmod
   - Checks device node creation

7. **Reporting Phase**
   - Generates rollback report
   - Documents restoration
   - Provides next steps
   - Creates validation checklist

**Rollback Validation:**
```bash
# After rollback, verify:

# 1. Modules restored
ls -la /lib/modules/$(uname -r)/extra/dsmil-72dev.ko
ls -la /lib/modules/$(uname -r)/kernel/drivers/tpm/tpm2_accel_early.ko

# 2. Modules loaded
lsmod | grep -E "dsmil|tpm2_accel"

# 3. Monitoring restored
ls -la /opt/dsmil/

# 4. Devices created
ls -la /dev/dsmil* /dev/tpm2*

# 5. No packages remain
dpkg -l | grep dell-milspec
```

**Exit Codes:**
- `0` - Rollback successful
- `1` - Rollback failed (check logs)

**Logs:**
- `/var/log/dell-milspec-rollback.log` - Detailed execution log

---

## Quick Start Guide

### First-Time Migration

```bash
# 1. Check current installation
cd /home/john/LAT5150DRVMIL/deployment
./detect-manual-install.sh --format text

# 2. Preview migration (no changes)
sudo ./migrate-to-packages.sh --dry-run --verbose

# 3. Perform migration
sudo ./migrate-to-packages.sh --verbose

# 4. Verify new installation
dsmil-status
milspec-control
```

### If Something Goes Wrong

```bash
# 1. Check logs
sudo tail -100 /var/log/dell-milspec-migration.log

# 2. Preview rollback
sudo ./rollback-migration.sh --dry-run --verbose

# 3. Perform rollback
sudo ./rollback-migration.sh --verbose

# 4. Verify manual installation restored
ls -la /opt/dsmil/
lsmod | grep dsmil
```

### Re-migration After Rollback

```bash
# You can migrate again anytime
sudo ./migrate-to-packages.sh --verbose
```

## Safety Features

### Comprehensive Backups
- Complete system state preserved
- Timestamped backup directories
- Compressed archives for long-term storage
- Detailed manifests

### Dry-Run Mode
- Test migrations without changes
- Validate commands and paths
- Preview all operations
- Safe for experimentation

### Validation at Every Step
- Pre-migration checks
- Post-migration validation
- Rollback verification
- Detailed error reporting

### Detailed Logging
- All operations logged
- Timestamps and exit codes
- Error messages preserved
- Debugging information

### User Confirmations
- Critical operations require confirmation
- Auto-confirm mode available for scripting
- Clear prompts and explanations

## Common Scenarios

### Scenario 1: Manual Installation → Packages

**Goal:** Migrate from manual install_dsmil_phase2a.sh installation to .deb packages.

```bash
sudo ./migrate-to-packages.sh --verbose
```

**Result:**
- Manual modules removed
- DKMS packages installed
- Configurations migrated
- Backup at `/var/backups/dell-milspec-manual-TIMESTAMP/`

### Scenario 2: Test Migration Without Changes

**Goal:** Preview migration to understand what will happen.

```bash
sudo ./migrate-to-packages.sh --dry-run --verbose
```

**Result:**
- No actual changes
- Complete preview of operations
- Log shows what would happen

### Scenario 3: Emergency Rollback

**Goal:** Something went wrong, restore manual installation.

```bash
sudo ./rollback-migration.sh --verbose
```

**Result:**
- Packages removed
- Manual installation restored
- System back to original state

### Scenario 4: Automated Migration (CI/CD)

**Goal:** Script-based migration without user interaction.

```bash
#!/bin/bash
set -e

# Non-interactive migration
sudo ./migrate-to-packages.sh --yes --verbose 2>&1 | tee migration.log

# Check result
if [ $? -eq 0 ]; then
    echo "Migration successful"
    # Run validation
    dsmil-status
else
    echo "Migration failed"
    # Automatic rollback
    sudo ./rollback-migration.sh --yes
fi
```

### Scenario 5: Multiple Systems

**Goal:** Migrate multiple Dell MIL-SPEC systems.

```bash
#!/bin/bash
# Deploy to multiple systems

SYSTEMS=("system1" "system2" "system3")

for sys in "${SYSTEMS[@]}"; do
    echo "Migrating $sys..."
    ssh root@$sys "cd /deployment && ./migrate-to-packages.sh --yes --verbose"

    # Verify
    if ssh root@$sys "dsmil-status"; then
        echo "$sys: SUCCESS"
    else
        echo "$sys: FAILED - rolling back"
        ssh root@$sys "cd /deployment && ./rollback-migration.sh --yes"
    fi
done
```

## Prerequisites

### System Requirements
- Dell Latitude 5450 MIL-SPEC (or compatible)
- Linux kernel 6.1+
- Debian/Ubuntu-based distribution
- Root privileges (sudo)

### Manual Installation Indicators
- `/opt/dsmil/` directory exists
- Kernel modules in `/lib/modules/*/extra/`
- Created by `install_dsmil_phase2a.sh` or `install_tpm2_module.sh`

### Package Requirements
- .deb packages available in apt repository or local
- `dell-milspec-tools_1.0.0-1_amd64.deb`
- `dell-milspec-dsmil-dkms_1.0.0-1_all.deb` (optional)
- `tpm2-accel-early-dkms_1.0.0-1_all.deb` (optional)

### Disk Space Requirements
- Backup: ~50-100 MB (depends on logs)
- Package installation: ~2-5 MB
- Total recommended free space: 200 MB

## Troubleshooting

### Migration Failed During Backup

**Symptom:** Script fails during backup phase.

**Solutions:**
```bash
# Check disk space
df -h /var/backups

# Skip backup (NOT RECOMMENDED)
sudo ./migrate-to-packages.sh --skip-backup --force

# Specify different backup location
sudo BACKUP_ROOT=/tmp/backup ./migrate-to-packages.sh
```

### Modules Won't Unload

**Symptom:** `modprobe -r` fails with "in use" error.

**Solutions:**
```bash
# Find what's using the module
lsmod | grep dsmil
sudo lsof | grep dsmil

# Force unload
sudo modprobe -rf dsmil-72dev

# Reboot and retry migration
sudo reboot
```

### Package Installation Failed

**Symptom:** `apt-get install` fails.

**Solutions:**
```bash
# Update package cache
sudo apt-get update

# Check package availability
apt-cache policy dell-milspec-tools

# Install from local .deb
sudo dpkg -i /path/to/dell-milspec-tools_1.0.0-1_amd64.deb
sudo apt-get install -f
```

### Rollback Can't Find Backup

**Symptom:** "No backups found" error.

**Solutions:**
```bash
# Search manually
sudo find /var/backups -name "dell-milspec-manual-*" -type d

# Specify backup path
sudo ./rollback-migration.sh --backup /var/backups/dell-milspec-manual-20251011_143022

# List backup contents
ls -la /var/backups/dell-milspec-manual-*/
```

### Configuration Migration Issues

**Symptom:** Configs not properly migrated.

**Solutions:**
```bash
# Check old configs in backup
cat /var/backups/dell-milspec-manual-*/configs/*

# Manually copy specific config
sudo cp /var/backups/.../configs/dsmil.json /etc/dell-milspec/

# Restore default configs
sudo cp /usr/share/dell-milspec/config/*.default /etc/dell-milspec/
```

## Advanced Usage

### Custom Backup Location

```bash
# Use different backup directory
export BACKUP_ROOT="/mnt/external/backups/dell-milspec-manual-$(date +%Y%m%d_%H%M%S)"
sudo -E ./migrate-to-packages.sh
```

### Selective Migration

```bash
# Migrate only specific components
# (requires script modification)

# Example: Keep manual monitoring, migrate modules
sudo ./migrate-to-packages.sh --modules-only
```

### Integration with Configuration Management

```bash
# Ansible playbook example
- name: Migrate to Dell MIL-SPEC packages
  hosts: dell_milspec_systems
  become: yes
  tasks:
    - name: Copy migration scripts
      copy:
        src: deployment/
        dest: /tmp/deployment/
        mode: '0755'

    - name: Detect manual installation
      command: /tmp/deployment/detect-manual-install.sh
      register: detection
      changed_when: false

    - name: Perform migration
      command: /tmp/deployment/migrate-to-packages.sh --yes
      when: detection.rc == 0
      register: migration

    - name: Verify installation
      command: dsmil-status
      changed_when: false
```

## File Sizes

```
migrate-to-packages.sh      26 KB    Main migration script
detect-manual-install.sh    14 KB    Detection utility
rollback-migration.sh       25 KB    Rollback script
README.md                   This file
```

## Version History

### 1.0.0 (2025-10-11)
- Initial release
- Complete migration workflow
- Comprehensive backup system
- Full rollback capability
- Production-ready error handling

## Security Considerations

### Privileged Operations
- Scripts require root/sudo
- Backup contains system files
- Configurations may contain sensitive data

### Backup Security
```bash
# Secure backup permissions
sudo chmod 700 /var/backups/dell-milspec-manual-*/
sudo chown root:root /var/backups/dell-milspec-manual-*/

# Encrypt backup
sudo tar czf - dell-milspec-manual-*/ | gpg -c > backup.tar.gz.gpg
```

### Audit Trail
- All operations logged
- Timestamps on all actions
- Before/after system state
- Rollback procedures documented

## Support

### Log Files
- `/var/log/dell-milspec-migration.log` - Migration log
- `/var/log/dell-milspec-rollback.log` - Rollback log
- `/var/backups/dell-milspec-manual-*/MANIFEST.txt` - Backup manifest
- `/var/backups/dell-milspec-manual-*/MIGRATION_REPORT.txt` - Migration report

### Debugging
```bash
# Enable verbose logging
export DEBUG=1
sudo -E ./migrate-to-packages.sh --verbose

# Check system state
./detect-manual-install.sh --format text --verbose
```

### Getting Help
```bash
# Show help for each script
./migrate-to-packages.sh --help
./detect-manual-install.sh --help
./rollback-migration.sh --help
```

## Best Practices

1. **Always test with --dry-run first**
   - Preview all changes
   - Understand impact
   - Validate scripts

2. **Keep backups**
   - Don't use --skip-backup
   - Archive to external storage
   - Test restoration

3. **Verify after migration**
   - Run dsmil-status
   - Test monitoring
   - Check device files

4. **Document customizations**
   - Note configuration changes
   - Record custom settings
   - Keep separate notes

5. **Plan rollback strategy**
   - Know backup location
   - Test rollback in lab
   - Have contingency plan

## License

GPL-3.0+ - See project LICENSE file

## Author

Claude Agent Framework v7.0 - DEPLOYER Agent
Specialized in deployment automation and migration

---

**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Last Updated:** 2025-10-11
**Version:** 1.0.0

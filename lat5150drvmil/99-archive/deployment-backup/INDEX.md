# Dell MIL-SPEC Platform - Deployment Directory Index

**Version:** 1.0.0
**Last Updated:** 2025-10-11
**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY

## Directory Purpose

Complete deployment automation and migration tooling for Dell Latitude 5450 MIL-SPEC platform, including package management, CI/CD integration, and manual-to-package migration scripts.

## Quick Navigation

### Migration Scripts (New - v1.0.0)
- **[migrate-to-packages.sh](./migrate-to-packages.sh)** - Main migration orchestrator
- **[detect-manual-install.sh](./detect-manual-install.sh)** - Manual installation detection
- **[rollback-migration.sh](./rollback-migration.sh)** - Emergency rollback script
- **[README.md](./README.md)** - Complete migration documentation

### Core Documentation
- **[FOUNDATION_COMPLETE.md](./FOUNDATION_COMPLETE.md)** - Deployment foundation overview
- **[IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md)** - Implementation details

### Subdirectories
- **[apt-repository/](./apt-repository/)** - Debian package repository
- **[debian-packages/](./debian-packages/)** - Built .deb packages
- **[ci/](./ci/)** - CI/CD integration scripts
- **[docs/](./docs/)** - Additional documentation
- **[man/](./man/)** - Manual pages
- **[runbooks/](./runbooks/)** - Operational runbooks
- **[scripts/](./scripts/)** - Utility scripts

## File Listing

### Migration Scripts (65 KB total)

#### migrate-to-packages.sh (26 KB)
**Purpose:** Main migration orchestrator for transitioning from manual installation to .deb packages

**Features:**
- Automatic manual installation detection
- Comprehensive backup creation with timestamps
- Safe removal of manual modules and files
- Package installation (tools, DKMS modules)
- Configuration migration and preservation
- Full validation and reporting
- Dry-run support

**Usage:**
```bash
sudo ./migrate-to-packages.sh --dry-run --verbose  # Preview
sudo ./migrate-to-packages.sh --verbose            # Execute
```

**Key Functions:**
- `detect_manual_installation()` - Scans for manual artifacts
- `create_backup()` - Creates timestamped backup
- `unload_manual_modules()` - Safely removes modules
- `remove_manual_files()` - Cleans up manual installation
- `install_packages()` - Installs .deb packages
- `migrate_configurations()` - Preserves user configs
- `validate_migration()` - Verifies new installation
- `generate_report()` - Creates detailed report

**Safety Features:**
- Pre-migration validation
- Comprehensive backups
- Step-by-step confirmation
- Detailed logging
- Rollback capability

**Exit Codes:**
- 0: Success
- 1: Failure

---

#### detect-manual-install.sh (14 KB)
**Purpose:** Detection utility for identifying manual installation artifacts

**Features:**
- Comprehensive artifact scanning
- Multiple output formats (JSON, text, summary)
- Exit codes for scripting
- Verbose debugging mode

**Usage:**
```bash
./detect-manual-install.sh                    # JSON output
./detect-manual-install.sh --format text      # Human-readable
./detect-manual-install.sh --format summary   # One-line summary
```

**Detection Areas:**
- Kernel modules (dsmil-72dev.ko, tpm2_accel_early.ko)
- Installation directory (/opt/dsmil/)
- Systemd services (dsmil-monitor.service)
- Configuration files (modprobe.d, modules-load.d)
- Manual tools (/usr/local/bin/)
- Device files (/dev/dsmil*, /dev/tpm2*)
- System groups (dsmil)

**Output Formats:**
```json
{
  "status": "manual_found",
  "summary": {
    "total_artifacts": 15,
    "modules": 2,
    "configs": 3,
    "services": 1
  }
}
```

**Exit Codes:**
- 0: Manual installation found
- 1: Clean system
- 2: Partial installation

---

#### rollback-migration.sh (25 KB)
**Purpose:** Emergency rollback from .deb packages to manual installation

**Features:**
- Automatic backup location and validation
- Safe package removal (purge)
- Complete restoration from backup
- Module reloading
- Configuration restoration
- Validation and reporting

**Usage:**
```bash
sudo ./rollback-migration.sh --dry-run --verbose  # Preview
sudo ./rollback-migration.sh --verbose            # Execute
sudo ./rollback-migration.sh --backup /path       # Specify backup
```

**Key Functions:**
- `find_backup()` - Locates and validates backup
- `detect_packages()` - Identifies installed packages
- `remove_packages()` - Purges .deb packages
- `restore_kernel_modules()` - Restores manual modules
- `restore_monitoring_system()` - Restores /opt/dsmil/
- `restore_configurations()` - Restores configs
- `validate_manual_installation()` - Verifies restoration
- `load_manual_modules()` - Loads restored modules

**Restoration Process:**
1. Locate backup in /var/backups/dell-milspec-manual-*
2. Validate backup integrity
3. Remove .deb packages completely
4. Restore kernel modules to /lib/modules/
5. Restore /opt/dsmil/ directory
6. Restore configuration files
7. Restore systemd service
8. Load manual modules
9. Validate restoration

**Exit Codes:**
- 0: Success
- 1: Failure

---

### Documentation (53 KB total)

#### README.md (19 KB)
**Purpose:** Complete migration documentation and user guide

**Contents:**
- Overview of all migration scripts
- Detailed feature descriptions
- Usage examples and scenarios
- Safety features
- Troubleshooting guide
- Best practices
- Quick start guide

**Sections:**
1. Migration Scripts Overview
2. Usage Instructions
3. Configuration Mapping
4. Common Scenarios
5. Troubleshooting
6. Advanced Usage
7. Security Considerations

---

#### FOUNDATION_COMPLETE.md (23 KB)
**Purpose:** Deployment foundation architecture and overview

**Contents:**
- Overall deployment strategy
- Package architecture
- Repository structure
- CI/CD integration
- Automation overview

---

#### IMPLEMENTATION_SUMMARY.md (11 KB)
**Purpose:** Implementation details and technical specifications

**Contents:**
- Technical implementation
- Package specifications
- DKMS integration
- Build process
- Testing procedures

---

## Directory Structure

```
deployment/
├── INDEX.md                          # This file
├── README.md                         # Migration documentation (19 KB)
├── FOUNDATION_COMPLETE.md            # Foundation overview (23 KB)
├── IMPLEMENTATION_SUMMARY.md         # Implementation details (11 KB)
│
├── Migration Scripts (65 KB)
│   ├── migrate-to-packages.sh        # Main migration (26 KB)
│   ├── detect-manual-install.sh      # Detection utility (14 KB)
│   └── rollback-migration.sh         # Rollback script (25 KB)
│
├── apt-repository/                   # Debian package repository
│   ├── conf/                         # Repository configuration
│   ├── db/                           # Repository database
│   ├── dists/                        # Distribution files
│   ├── pool/                         # Package pool
│   └── README.md                     # Repository documentation
│
├── debian-packages/                  # Built .deb packages
│   ├── dell-milspec-tools/           # Tools package source
│   ├── dell-milspec-dsmil-dkms/      # DSMIL DKMS source
│   ├── tpm2-accel-early-dkms/        # TPM2 DKMS source
│   └── *.deb                         # Built packages
│
├── ci/                               # CI/CD integration
│   ├── github-actions/               # GitHub Actions workflows
│   ├── gitlab-ci/                    # GitLab CI configuration
│   └── jenkins/                      # Jenkins pipelines
│
├── docs/                             # Additional documentation
│   ├── QUICKSTART.md                 # Quick start guide
│   ├── ADVANCED_USAGE.md             # Advanced usage
│   └── TROUBLESHOOTING.md            # Troubleshooting
│
├── man/                              # Manual pages
│   ├── man1/                         # User commands
│   ├── man5/                         # Configuration files
│   ├── man7/                         # Overviews
│   └── man8/                         # System commands
│
├── runbooks/                         # Operational runbooks
│   ├── deployment.md                 # Deployment procedures
│   ├── maintenance.md                # Maintenance tasks
│   └── troubleshooting.md            # Issue resolution
│
└── scripts/                          # Utility scripts
    ├── build-packages.sh             # Build all packages
    ├── test-packages.sh              # Test packages
    └── update-repository.sh          # Update apt repository
```

## Usage Workflows

### Workflow 1: First-Time Migration

```bash
# 1. Navigate to deployment directory
cd /home/john/LAT5150DRVMIL/deployment

# 2. Check for manual installation
./detect-manual-install.sh --format text

# 3. Preview migration (no changes)
sudo ./migrate-to-packages.sh --dry-run --verbose

# 4. Perform migration
sudo ./migrate-to-packages.sh --verbose

# 5. Verify installation
dsmil-status
milspec-control
```

**Expected Result:**
- Manual installation backed up to `/var/backups/dell-milspec-manual-TIMESTAMP/`
- .deb packages installed
- Configurations migrated to `/etc/dell-milspec/`
- New commands available: `dsmil-status`, `milspec-control`, etc.

---

### Workflow 2: Emergency Rollback

```bash
# 1. Navigate to deployment directory
cd /home/john/LAT5150DRVMIL/deployment

# 2. Preview rollback
sudo ./rollback-migration.sh --dry-run --verbose

# 3. Perform rollback
sudo ./rollback-migration.sh --verbose

# 4. Verify manual installation
ls -la /opt/dsmil/
lsmod | grep dsmil
```

**Expected Result:**
- .deb packages removed
- Manual installation restored from backup
- Modules loaded from `/lib/modules/*/extra/`
- `/opt/dsmil/` directory restored

---

### Workflow 3: Clean System Package Installation

```bash
# If no manual installation exists, install packages directly

# 1. Detect (should return "clean")
./detect-manual-install.sh --format summary

# 2. Install from repository
sudo apt-get update
sudo apt-get install dell-milspec-tools

# 3. Optional: Install DKMS modules
sudo apt-get install dell-milspec-dsmil-dkms tpm2-accel-early-dkms

# 4. Verify
dsmil-status
```

---

### Workflow 4: CI/CD Automated Deployment

```bash
# Example automated deployment script

#!/bin/bash
set -e

# Build packages
cd /deployment
./scripts/build-packages.sh

# Test packages
./scripts/test-packages.sh

# Update repository
./scripts/update-repository.sh

# Deploy to target systems
ansible-playbook deploy-milspec-packages.yml
```

---

## Migration States

### State 1: Manual Installation (Pre-Migration)

**Characteristics:**
- `/opt/dsmil/` directory exists
- Modules in `/lib/modules/*/extra/`
- Manual configuration files
- Created by `install_dsmil_phase2a.sh`

**Detection:**
```bash
./detect-manual-install.sh
# Exit code: 0 (manual_found)
```

---

### State 2: Transitioning (During Migration)

**Characteristics:**
- Backup being created
- Old files being removed
- New packages being installed
- Configurations being migrated

**Status:**
```bash
# Check migration log
sudo tail -f /var/log/dell-milspec-migration.log
```

---

### State 3: Package-Based (Post-Migration)

**Characteristics:**
- `/etc/dell-milspec/` configurations
- DKMS-managed modules
- System commands available
- Package manager controls

**Verification:**
```bash
dpkg -l | grep dell-milspec
dsmil-status
milspec-control
```

---

### State 4: Rolled Back (After Rollback)

**Characteristics:**
- Back to manual installation
- Packages removed
- `/opt/dsmil/` restored
- Manual modules loaded

**Verification:**
```bash
ls -la /opt/dsmil/
lsmod | grep dsmil
dpkg -l | grep dell-milspec  # Should be empty
```

---

## Configuration Mapping

### Manual Installation → Package Installation

| Component | Manual Location | Package Location | Notes |
|-----------|----------------|------------------|-------|
| **DSMIL Module** | `/lib/modules/*/extra/dsmil-72dev.ko` | DKMS managed | Auto-rebuilt per kernel |
| **TPM2 Module** | `/lib/modules/*/kernel/drivers/tpm/tpm2_accel_early.ko` | DKMS managed | Auto-rebuilt per kernel |
| **Monitoring Scripts** | `/opt/dsmil/monitoring/*.py` | `/usr/share/dell-milspec/monitoring/*.py` | System-wide location |
| **Configuration Files** | `/opt/dsmil/config/*.json` | `/etc/dell-milspec/*.{conf,json}` | FHS-compliant |
| **User Commands** | `/opt/dsmil/bin/*` | `/usr/bin/dsmil-*`, `/usr/bin/milspec-*` | In PATH |
| **System Commands** | None | `/usr/sbin/milspec-*` | Admin commands |
| **Log Files** | `/opt/dsmil/logs/` | `/var/log/dell-milspec/` | Standard location |
| **Systemd Service** | `/etc/systemd/system/dsmil-monitor.service` | Removed (manual start) | Use `milspec-monitor` |
| **Modprobe Config** | `/etc/modprobe.d/dsmil-72dev.conf` | `/etc/modprobe.d/dell-milspec.conf` | Unified config |
| **Device Permissions** | Manual via udev | Handled by package | Auto-configured |

---

## Backup Structure

When migration creates a backup, it follows this structure:

```
/var/backups/dell-milspec-manual-YYYYMMDD_HHMMSS/
├── MANIFEST.txt                      # Complete inventory
├── MIGRATION_REPORT.txt              # Migration results
│
├── modules/                          # Kernel modules
│   ├── dsmil-72dev.ko
│   └── tpm2_accel_early.ko
│
├── monitoring/                       # Monitoring system
│   └── dsmil/                        # Complete /opt/dsmil/ copy
│       ├── monitoring/
│       │   ├── dsmil_comprehensive_monitor.py
│       │   └── safe_token_tester.py
│       ├── config/
│       │   ├── dsmil.json
│       │   ├── monitoring.json
│       │   └── safety.json
│       ├── logs/
│       └── bin/
│
├── configs/                          # Configuration files
│   ├── dsmil-72dev.conf
│   ├── tpm2-acceleration.conf
│   └── *.rules
│
├── services/                         # Systemd services
│   └── dsmil-monitor.service
│
└── logs/                             # Historical logs
    └── *.log
```

**Backup Archive:**
- Compressed: `/var/backups/dell-milspec-manual-YYYYMMDD_HHMMSS.tar.gz`
- Size: ~50-100 MB (depends on logs)
- Retention: Manual (not auto-deleted)

---

## Logs and Reports

### Migration Logs

**Location:** `/var/log/dell-milspec-migration.log`

**Contents:**
- Timestamped operations
- Command execution
- Success/failure status
- Error messages
- Validation results

**Example:**
```
[2025-10-11 15:30:00] Migration to .deb packages started
[2025-10-11 15:30:05] INFO: Manual installation detected
[2025-10-11 15:30:10] SUCCESS: Backup created
[2025-10-11 15:30:45] SUCCESS: Packages installed
[2025-10-11 15:31:00] SUCCESS: Migration completed
```

---

### Rollback Logs

**Location:** `/var/log/dell-milspec-rollback.log`

**Contents:**
- Backup validation
- Package removal
- File restoration
- Module loading
- Validation results

---

### Migration Report

**Location:** `/var/backups/dell-milspec-manual-TIMESTAMP/MIGRATION_REPORT.txt`

**Contents:**
- System information
- Migration summary
- Installed packages
- Configuration changes
- Next steps
- Rollback instructions

---

## Security Considerations

### Privileged Operations
- All migration scripts require root/sudo
- Backups contain system files
- Configurations may have sensitive data

### Backup Security
```bash
# Secure backup permissions
sudo chmod 700 /var/backups/dell-milspec-manual-*/
sudo chown root:root /var/backups/dell-milspec-manual-*/

# Encrypt backup for archival
sudo tar czf - dell-milspec-manual-*/ | \
    gpg --symmetric --cipher-algo AES256 > backup.tar.gz.gpg
```

### Audit Trail
- All operations logged with timestamps
- Before/after system state captured
- Rollback procedures documented
- Changes traceable

---

## Prerequisites

### For Migration
- Manual installation present (detectable)
- Root/sudo privileges
- Sufficient disk space (~200 MB)
- .deb packages available

### For Rollback
- Valid backup directory
- Root/sudo privileges
- Packages currently installed

### For Clean Install
- Clean system (no manual install)
- .deb packages available
- Root/sudo privileges

---

## Troubleshooting

### Issue: "No manual installation detected"

**Cause:** System is already clean or using packages.

**Solution:**
```bash
# Check what's installed
dpkg -l | grep dell-milspec

# If packages exist, no migration needed
# If nothing exists, install packages directly
sudo apt-get install dell-milspec-tools
```

---

### Issue: "Backup failed - disk space"

**Cause:** Insufficient space in /var/backups.

**Solution:**
```bash
# Check space
df -h /var/backups

# Use different backup location
export BACKUP_ROOT="/mnt/external/backup-$(date +%Y%m%d_%H%M%S)"
sudo -E ./migrate-to-packages.sh
```

---

### Issue: "Module won't unload - in use"

**Cause:** Something is using the module.

**Solution:**
```bash
# Find what's using it
lsmod | grep dsmil
sudo lsof | grep dsmil

# Kill processes
sudo pkill -f dsmil

# Force unload
sudo modprobe -rf dsmil-72dev

# Or reboot
sudo reboot
```

---

### Issue: "Rollback can't find backup"

**Cause:** Backup directory not in search path.

**Solution:**
```bash
# Find backup manually
sudo find /var/backups -name "dell-milspec-manual-*" -type d

# Specify path
sudo ./rollback-migration.sh --backup /var/backups/dell-milspec-manual-20251011_143022
```

---

## Best Practices

1. **Always preview first**
   ```bash
   sudo ./migrate-to-packages.sh --dry-run --verbose
   ```

2. **Keep backups safe**
   - Don't delete backup directories
   - Archive to external storage
   - Test restoration periodically

3. **Verify after migration**
   ```bash
   dsmil-status
   milspec-control
   lsmod | grep dsmil
   ```

4. **Document customizations**
   - Note any custom configurations
   - Keep separate records
   - Include in backup notes

5. **Test rollback in lab first**
   - Understand the process
   - Verify timing
   - Document issues

---

## Quick Reference

### Common Commands

```bash
# Detect manual installation
./detect-manual-install.sh --format text

# Preview migration
sudo ./migrate-to-packages.sh --dry-run

# Perform migration
sudo ./migrate-to-packages.sh --verbose

# Preview rollback
sudo ./rollback-migration.sh --dry-run

# Perform rollback
sudo ./rollback-migration.sh --verbose

# Check logs
sudo tail -f /var/log/dell-milspec-migration.log
sudo tail -f /var/log/dell-milspec-rollback.log

# Verify installation
dsmil-status
milspec-control
dpkg -l | grep dell-milspec
```

---

## Version History

### 1.0.0 (2025-10-11)
- Initial release
- Complete migration workflow
- Comprehensive backup system
- Full rollback capability
- Production-ready error handling
- Detailed documentation

---

## Support

### Documentation
- **[README.md](./README.md)** - Complete migration guide
- **[FOUNDATION_COMPLETE.md](./FOUNDATION_COMPLETE.md)** - Architecture overview
- **[IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md)** - Technical details

### Logs
- `/var/log/dell-milspec-migration.log`
- `/var/log/dell-milspec-rollback.log`
- `/var/backups/dell-milspec-manual-*/MANIFEST.txt`

### Help Commands
```bash
./migrate-to-packages.sh --help
./detect-manual-install.sh --help
./rollback-migration.sh --help
```

---

## License

GPL-3.0+ - See project LICENSE file

## Author

Claude Agent Framework v7.0 - DEPLOYER Agent
Specialized in deployment automation and migration

---

**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Last Updated:** 2025-10-11
**Version:** 1.0.0

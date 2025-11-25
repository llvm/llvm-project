#!/bin/bash
# JANITOR Agent - Root Directory Cleanup Script
# LAT5150DRVMIL Root Directory Cleanup
# Generated: 2025-10-11
# Agent: JANITOR from Claude Agent Framework v7.0
#
# Purpose: Clean root directory and move files to appropriate locations
# Safety: Includes dry-run mode and backup creation

set -eo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/home/john/LAT5150DRVMIL"
DRY_RUN="${DRY_RUN:-1}"  # Default to dry-run mode
BACKUP_DIR="${PROJECT_ROOT}/99-archive/root-cleanup-backup-$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${PROJECT_ROOT}/root-cleanup-$(date +%Y%m%d-%H%M%S).log"

# Statistics counters
FILES_DELETED=0
FILES_MOVED=0
DIRS_DELETED=0
BYTES_FREED=0

# Logging function
log() {
    echo -e "${2:-$NC}$1${NC}" | tee -a "$LOG_FILE"
}

# Header
log "================================================" "$BLUE"
log "JANITOR AGENT - Root Directory Cleanup" "$BLUE"
log "================================================" "$BLUE"
log "Project: LAT5150DRVMIL"
log "Date: $(date)"
log "Mode: $([ "$DRY_RUN" = "1" ] && echo 'DRY-RUN' || echo 'LIVE')"
log "Backup: $BACKUP_DIR"
log ""

# Safety check
if [ "$DRY_RUN" = "0" ]; then
    log "WARNING: Running in LIVE mode. Files will be deleted/moved!" "$YELLOW"
    log "Press Ctrl+C to cancel, or Enter to continue..." "$YELLOW"
    read -r
fi

# Create backup directory structure
mkdir -p "$BACKUP_DIR"/{files,scripts,source,config,logs,data}

# Function to safely delete file
safe_delete() {
    local file="$1"
    local reason="$2"

    if [ ! -f "$file" ]; then
        return
    fi

    local size=$(stat -c "%s" "$file" 2>/dev/null || echo "0")
    local basename=$(basename "$file")

    log "DELETE: $basename ($reason)" "$RED"

    if [ "$DRY_RUN" = "0" ]; then
        cp "$file" "$BACKUP_DIR/files/"
        rm "$file"
        FILES_DELETED=$((FILES_DELETED + 1))
        BYTES_FREED=$((BYTES_FREED + ${size:-0}))
    fi
}

# Function to safely move file
safe_move() {
    local file="$1"
    local dest="$2"
    local reason="$3"

    if [ ! -f "$file" ]; then
        return
    fi

    local basename=$(basename "$file")

    log "MOVE: $basename -> $dest ($reason)" "$GREEN"

    if [ "$DRY_RUN" = "0" ]; then
        cp "$file" "$BACKUP_DIR/files/"
        mkdir -p "$(dirname "$dest")"
        mv "$file" "$dest"
        FILES_MOVED=$((FILES_MOVED + 1))
    fi
}

# Function to safely delete directory
safe_delete_dir() {
    local dir="$1"
    local reason="$2"

    if [ ! -d "$dir" ]; then
        return
    fi

    local dirname=$(basename "$dir")
    local size=$(du -sh "$dir" | cut -f1)

    log "DELETE DIR: $dirname [$size] ($reason)" "$RED"

    if [ "$DRY_RUN" = "0" ]; then
        cp -r "$dir" "$BACKUP_DIR/"
        rm -rf "$dir"
        DIRS_DELETED=$((DIRS_DELETED + 1))
    fi
}

cd "$PROJECT_ROOT"

log "=== PHASE 1: Remove Old Documentation Backups ===" "$BLUE"
log "Found 5 documentation backup directories (total ~20KB)"
for dir in documentation_backup_*; do
    [ -d "$dir" ] && safe_delete_dir "$dir" "Old DOCGEN backup - no longer needed"
done

log ""
log "=== PHASE 2: Remove Old Baseline Snapshots ===" "$BLUE"
safe_delete_dir "baseline_20250901_024258" "Old baseline snapshot"
safe_delete_dir "baseline_20250901_024305" "Old baseline snapshot"
safe_delete "baseline_20250901_024258.tar.gz" "Old baseline archive"
safe_delete "baseline_20250901_024305.tar.gz" "Old baseline archive"

log ""
log "=== PHASE 3: Remove Old Enumeration Data ===" "$BLUE"
safe_delete_dir "dsmil_enumeration_20250815_142321" "Old enumeration data"
safe_delete "dsmil_token_discovery.txt" "Old token discovery data"
safe_delete "dsmil_token_discovery_20250901_025222.txt" "Old token discovery data"
safe_delete "dsmil_token_analysis_report.txt" "Old token analysis"
safe_delete "token_analysis.txt" "Old token analysis"
safe_delete "all_tokens_dump.txt" "Old token dump"

log ""
log "=== PHASE 4: Remove Mock/Test Backup Directories ===" "$BLUE"
safe_delete_dir "mock_backup_20250902-202546" "Mock backup directory"
safe_delete_dir "mock_backup_20250902-202859" "Mock backup directory"
safe_delete_dir "mock_monitoring" "Mock monitoring directory"

log ""
log "=== PHASE 5: Remove Old Build Artifacts ===" "$BLUE"
safe_delete_dir "__pycache__" "Python cache directory"
safe_delete_dir "obj" "Old object file directory"
safe_delete_dir "bin" "Old binary directory - conflicts with organized structure"

log ""
log "=== PHASE 6: Move Compilation Binaries to 99-archive ===" "$BLUE"
# Compiled test binaries
safe_move "test_device_access" "$BACKUP_DIR/files/test_device_access" "Compiled test binary"
safe_move "test_ioctl_simple" "$BACKUP_DIR/files/test_ioctl_simple" "Compiled test binary"
safe_move "test_phase1_discovery" "$BACKUP_DIR/files/test_phase1_discovery" "Compiled test binary"
safe_move "test_simple_ioctl" "$BACKUP_DIR/files/test_simple_ioctl" "Compiled test binary"
safe_move "test_simd" "$BACKUP_DIR/files/test_simd" "Compiled test binary"
safe_move "ioctl_probe_correct" "$BACKUP_DIR/files/ioctl_probe_correct" "Compiled probe binary"
safe_move "ioctl_probe_simple" "$BACKUP_DIR/files/ioctl_probe_simple" "Compiled probe binary"

log ""
log "=== PHASE 7: Move C Source Files to 01-source ===" "$BLUE"
# Move all .c and .h files to source directory
for file in *.c; do
    [ -f "$file" ] && safe_move "$file" "${PROJECT_ROOT}/01-source/tests/$file" "C source file belongs in source tree"
done

for file in *.h; do
    [ -f "$file" ] && safe_move "$file" "${PROJECT_ROOT}/01-source/kernel/$file" "C header file belongs in source tree"
done

log ""
log "=== PHASE 8: Move Build System Files ===" "$BLUE"
safe_move "Makefile" "${PROJECT_ROOT}/01-source/Makefile" "Main Makefile"
safe_move "Makefile.avx" "${PROJECT_ROOT}/01-source/Makefile.avx" "AVX-specific Makefile"
safe_move "Makefile.probe" "${PROJECT_ROOT}/01-source/Makefile.probe" "Probe-specific Makefile"

log ""
log "=== PHASE 9: Archive Deployment Scripts ===" "$BLUE"
# Installation scripts
safe_move "install_dsmil_phase2a.sh" "${PROJECT_ROOT}/02-deployment/scripts/install_dsmil_phase2a.sh" "Deployment script"
safe_move "install_dsmil_phase2a_integrated.sh" "${PROJECT_ROOT}/02-deployment/scripts/install_dsmil_phase2a_integrated.sh" "Deployment script"
safe_move "install_thermal_guardian.sh" "${PROJECT_ROOT}/02-deployment/scripts/install_thermal_guardian.sh" "Deployment script"
safe_move "activate_phase1_production.sh" "${PROJECT_ROOT}/02-deployment/scripts/activate_phase1_production.sh" "Deployment script"

# Test and utility scripts
safe_move "test_installer.sh" "${PROJECT_ROOT}/01-source/tests/test_installer.sh" "Test script"
safe_move "test_dsmil_smi_access.sh" "${PROJECT_ROOT}/01-source/tests/test_dsmil_smi_access.sh" "Test script"
safe_move "test_military_devices.sh" "${PROJECT_ROOT}/01-source/tests/test_military_devices.sh" "Test script"
safe_move "testing-environment-config.sh" "${PROJECT_ROOT}/01-source/tests/testing-environment-config.sh" "Test configuration"

# Discovery and monitoring scripts
safe_move "discover_dsmil_tokens.sh" "${PROJECT_ROOT}/01-source/scripts/discover_dsmil_tokens.sh" "Discovery script"
safe_move "launch_dsmil_monitor.sh" "${PROJECT_ROOT}/02-deployment/scripts/launch_dsmil_monitor.sh" "Monitor script"
safe_move "load_dsmil_module.sh" "${PROJECT_ROOT}/02-deployment/scripts/load_dsmil_module.sh" "Module loading script"

# System scripts
safe_move "fix_library_path.sh" "${PROJECT_ROOT}/02-deployment/scripts/fix_library_path.sh" "System fix script"
safe_move "validate-system-safety.sh" "${PROJECT_ROOT}/02-deployment/scripts/validate-system-safety.sh" "Validation script"
safe_move "create-baseline-snapshot.sh" "${PROJECT_ROOT}/02-deployment/scripts/create-baseline-snapshot.sh" "Backup script"

# Documentation and utility scripts
safe_move "launch-docs-browser.sh" "${PROJECT_ROOT}/00-documentation/scripts/launch-docs-browser.sh" "Documentation utility"
safe_move "view-dsmil-docs.sh" "${PROJECT_ROOT}/00-documentation/scripts/view-dsmil-docs.sh" "Documentation utility"

# Thermal and token scripts
safe_move "quick_thermal_test.sh" "${PROJECT_ROOT}/01-source/tests/quick_thermal_test.sh" "Thermal test"
safe_move "wmi_token_control.sh" "${PROJECT_ROOT}/01-source/scripts/wmi_token_control.sh" "Token control"
safe_move "run_ioctl_discovery.sh" "${PROJECT_ROOT}/01-source/scripts/run_ioctl_discovery.sh" "Discovery script"
safe_move "start-monitor.sh" "${PROJECT_ROOT}/02-deployment/scripts/start-monitor.sh" "Monitor startup"

log ""
log "=== PHASE 10: Archive Python Scripts ===" "$BLUE"
# Deployment scripts
for script in deploy_*.py *_deployment*.py; do
    [ -f "$script" ] && safe_move "$script" "${PROJECT_ROOT}/02-deployment/scripts/$script" "Deployment Python script"
done

# Test scripts
for script in test_*.py validate_*.py; do
    [ -f "$script" ] && safe_move "$script" "${PROJECT_ROOT}/01-source/tests/$script" "Test Python script"
done

# Analysis and reconnaissance scripts
for script in analyze_*.py investigate_*.py explore_*.py nsa_*.py; do
    [ -f "$script" ] && safe_move "$script" "${PROJECT_ROOT}/01-source/debugging/$script" "Analysis script"
done

# Monitoring and dashboard scripts
for script in *_monitor*.py *_dashboard*.py *_status.py; do
    [ -f "$script" ] && safe_move "$script" "${PROJECT_ROOT}/02-deployment/monitoring/$script" "Monitoring script"
done

# Token and activation scripts
for script in activate_*.py begin_token_*.py *_token*.py; do
    [ -f "$script" ] && safe_move "$script" "${PROJECT_ROOT}/01-source/scripts/$script" "Token management script"
done

# Emergency and fix scripts
for script in *_emergency*.py fix_*.py *_fixes*.py safe_*.py; do
    [ -f "$script" ] && safe_move "$script" "${PROJECT_ROOT}/02-deployment/scripts/$script" "Emergency/fix script"
done

# Orchestration scripts
for script in *orchestrat*.py agent_task_bridge.py comprehensive_agent_*.py; do
    [ -f "$script" ] && safe_move "$script" "${PROJECT_ROOT}/02-deployment/orchestration/$script" "Orchestration script"
done

# Remaining Python files (demos, utilities, etc.)
for script in *.py; do
    [ -f "$script" ] && safe_move "$script" "${PROJECT_ROOT}/01-source/scripts/$script" "Python utility script"
done

log ""
log "=== PHASE 11: Archive Configuration and Data Files ===" "$BLUE"
# Config files
for file in *.conf *.env *.service; do
    [ -f "$file" ] && safe_move "$file" "${PROJECT_ROOT}/02-deployment/config/$file" "Configuration file"
done

# Log files
for file in *.log; do
    [ -f "$file" ] && safe_move "$file" "${PROJECT_ROOT}/99-archive/old-logs/$file" "Old log file"
done

# JSON data files
for file in *.json; do
    [ -f "$file" ] && safe_move "$file" "${PROJECT_ROOT}/99-archive/old-data/$file" "Old data file"
done

# Patch files
for file in *.patch; do
    [ -f "$file" ] && safe_move "$file" "${PROJECT_ROOT}/01-source/patches/$file" "Patch file"
done

# Text data files
for file in *.txt; do
    [ -f "$file" ] && safe_move "$file" "${PROJECT_ROOT}/99-archive/old-data/$file" "Old text data"
done

# Archive files
for file in *.tar.gz *.tgz *.zip; do
    [ -f "$file" ] && safe_move "$file" "${PROJECT_ROOT}/99-archive/old-backups/$file" "Old archive"
done

log ""
log "=== PHASE 12: Archive Old Environment Directories ===" "$BLUE"
safe_delete_dir "LAT5150_DEV" "Old development environment - superseded by organized structure"
safe_delete_dir "LAT5150_PROD" "Old production environment - superseded by organized structure"

log ""
log "=== PHASE 13: Remove Duplicate/Redundant Directories ===" "$BLUE"
# Check for duplicate directories
if [ -d "deployment" ] && [ -d "02-deployment" ]; then
    log "WARNING: Both 'deployment' and '02-deployment' exist" "$YELLOW"
    log "Manual review recommended before deleting 'deployment/'" "$YELLOW"
fi

if [ -d "docs" ] && [ -d "00-documentation" ]; then
    log "WARNING: Both 'docs' and '00-documentation' exist" "$YELLOW"
    log "Manual review recommended before deleting 'docs/'" "$YELLOW"
fi

if [ -d "scripts" ] && [ -d "02-deployment/scripts" ]; then
    safe_delete_dir "scripts" "Superseded by organized 02-deployment/scripts"
fi

# Archive old directories that may have unique content
for dir in deployment_monitoring logs monitoring docu assets database infrastructure testing thermal-guardian web-interface; do
    if [ -d "$dir" ]; then
        log "ARCHIVE: $dir -> 99-archive/$dir (may contain unique content)" "$YELLOW"
        if [ "$DRY_RUN" = "0" ]; then
            mkdir -p "${PROJECT_ROOT}/99-archive/"
            mv "$dir" "${PROJECT_ROOT}/99-archive/"
            DIRS_DELETED=$((DIRS_DELETED + 1))
        fi
    fi
done

log ""
log "================================================" "$BLUE"
log "CLEANUP SUMMARY" "$BLUE"
log "================================================" "$BLUE"
log "Files Deleted: $FILES_DELETED"
log "Files Moved: $FILES_MOVED"
log "Directories Deleted: $DIRS_DELETED"
log "Space Freed: $(numfmt --to=iec-i --suffix=B $BYTES_FREED 2>/dev/null || echo "$BYTES_FREED bytes")"
log ""
log "Backup Location: $BACKUP_DIR"
log "Log File: $LOG_FILE"
log ""

if [ "$DRY_RUN" = "1" ]; then
    log "=== DRY-RUN MODE - No changes were made ===" "$YELLOW"
    log "To execute cleanup, run:" "$YELLOW"
    log "  DRY_RUN=0 $0" "$YELLOW"
else
    log "=== CLEANUP COMPLETED SUCCESSFULLY ===" "$GREEN"
fi

log ""
log "Next steps:" "$BLUE"
log "1. Review backup in: $BACKUP_DIR"
log "2. Test project functionality"
log "3. Review and update .gitignore"
log "4. Commit cleanup changes to git"
log ""

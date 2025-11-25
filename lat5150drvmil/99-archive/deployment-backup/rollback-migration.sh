#!/bin/bash
#
# Dell MIL-SPEC Platform - Migration Rollback Script
# Rolls back from .deb packages to manual installation
#
# Version: 1.0.0
# Author: Claude Agent Framework - DEPLOYER Agent
# Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
#

set -euo pipefail

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

SCRIPT_VERSION="1.0.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Backup search locations
BACKUP_SEARCH_PATHS=(
    "/var/backups/dell-milspec-manual-*"
    "$(pwd)/dell-milspec-manual-*"
)

# Package names
DSMIL_PACKAGE="dell-milspec-dsmil-dkms"
TPM2_PACKAGE="tpm2-accel-early-dkms"
TOOLS_PACKAGE="dell-milspec-tools"

# Installation paths
MANUAL_INSTALL_DIR="/opt/dsmil"
MANUAL_SERVICE="/etc/systemd/system/dsmil-monitor.service"

# Rollback log
ROLLBACK_LOG="/var/log/dell-milspec-rollback.log"

# Flags
DRY_RUN=false
FORCE_MODE=false
AUTO_CONFIRM=false
VERBOSE=false
BACKUP_PATH=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# =============================================================================
# LOGGING AND OUTPUT FUNCTIONS
# =============================================================================

init_logging() {
    local log_dir=$(dirname "$ROLLBACK_LOG")
    sudo mkdir -p "$log_dir"
    sudo touch "$ROLLBACK_LOG"
    sudo chmod 644 "$ROLLBACK_LOG"

    log "Rollback from .deb packages started"
    log "Version: $SCRIPT_VERSION"
    log "Timestamp: $TIMESTAMP"
    log "Dry run: $DRY_RUN"
}

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" | sudo tee -a "$ROLLBACK_LOG" > /dev/null
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${CYAN}${msg}${NC}"
    fi
}

info() {
    echo -e "${BLUE}[INFO]${NC} $*"
    log "INFO: $*"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
    log "SUCCESS: $*"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
    log "WARNING: $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
    log "ERROR: $*"
}

critical() {
    echo -e "${RED}${BOLD}[CRITICAL]${NC} $*" >&2
    log "CRITICAL: $*"
}

section() {
    echo
    echo -e "${BOLD}${PURPLE}========================================================================${NC}"
    echo -e "${BOLD}${PURPLE} $*${NC}"
    echo -e "${BOLD}${PURPLE}========================================================================${NC}"
    echo
    log "SECTION: $*"
}

progress_bar() {
    local current=$1
    local total=$2
    local desc=$3
    local percent=$((current * 100 / total))
    local filled=$((percent / 2))
    local empty=$((50 - filled))

    printf "\r${CYAN}[%3d%%]${NC} [" "$percent"
    printf "%${filled}s" | tr ' ' '█'
    printf "%${empty}s" | tr ' ' '░'
    printf "] %s" "$desc"

    if [[ $current -eq $total ]]; then
        echo
    fi
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run with sudo privileges"
        echo "Please run: sudo $0 $*"
        exit 1
    fi
}

confirm() {
    local prompt="$1"
    local default="${2:-n}"

    if [[ "$AUTO_CONFIRM" == "true" ]]; then
        return 0
    fi

    if [[ "$default" == "y" ]]; then
        read -p "$prompt [Y/n]: " -r response
        response=${response:-y}
    else
        read -p "$prompt [y/N]: " -r response
        response=${response:-n}
    fi

    [[ "$response" =~ ^[Yy]$ ]]
}

execute() {
    local cmd="$1"
    local description="$2"

    if [[ "$VERBOSE" == "true" ]]; then
        info "Executing: $cmd"
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would execute: $cmd"
        return 0
    fi

    if eval "$cmd" >> "$ROLLBACK_LOG" 2>&1; then
        success "$description"
        return 0
    else
        error "$description failed"
        return 1
    fi
}

# =============================================================================
# BACKUP DETECTION
# =============================================================================

find_backup() {
    section "LOCATING BACKUP"

    if [[ -n "$BACKUP_PATH" ]] && [[ -d "$BACKUP_PATH" ]]; then
        info "Using specified backup: $BACKUP_PATH"
        return 0
    fi

    info "Searching for backups..."

    local backups=()
    for pattern in "${BACKUP_SEARCH_PATHS[@]}"; do
        while IFS= read -r -d '' backup; do
            if [[ -d "$backup" ]] && [[ -f "$backup/MANIFEST.txt" ]]; then
                backups+=("$backup")
            fi
        done < <(find $(dirname "$pattern") -maxdepth 1 -type d -name "$(basename "$pattern")" -print0 2>/dev/null || true)
    done

    if [[ ${#backups[@]} -eq 0 ]]; then
        error "No backups found"
        echo "Searched locations:"
        for pattern in "${BACKUP_SEARCH_PATHS[@]}"; do
            echo "  - $pattern"
        done
        return 1
    fi

    # Sort backups by timestamp (newest first)
    IFS=$'\n' backups=($(sort -r <<<"${backups[*]}"))

    if [[ ${#backups[@]} -eq 1 ]]; then
        BACKUP_PATH="${backups[0]}"
        success "Found backup: $BACKUP_PATH"
    else
        echo "Multiple backups found:"
        for i in "${!backups[@]}"; do
            echo "  $((i+1)). ${backups[$i]}"
        done

        if [[ "$AUTO_CONFIRM" == "false" ]]; then
            read -p "Select backup to restore [1]: " -r selection
            selection=${selection:-1}
            BACKUP_PATH="${backups[$((selection-1))]}"
        else
            BACKUP_PATH="${backups[0]}"
            warning "Auto-selecting most recent backup: $BACKUP_PATH"
        fi

        success "Selected backup: $BACKUP_PATH"
    fi

    # Validate backup
    validate_backup
}

validate_backup() {
    info "Validating backup..."

    local required_files=(
        "MANIFEST.txt"
    )

    for file in "${required_files[@]}"; do
        if [[ ! -f "$BACKUP_PATH/$file" ]]; then
            error "Backup validation failed: missing $file"
            return 1
        fi
    done

    # Check for backed up components
    local has_modules=false
    local has_monitoring=false

    if [[ -d "$BACKUP_PATH/modules" ]] && [[ -n "$(ls -A $BACKUP_PATH/modules 2>/dev/null)" ]]; then
        has_modules=true
    fi

    if [[ -d "$BACKUP_PATH/monitoring" ]] && [[ -n "$(ls -A $BACKUP_PATH/monitoring 2>/dev/null)" ]]; then
        has_monitoring=true
    fi

    if [[ "$has_modules" == "false" ]] && [[ "$has_monitoring" == "false" ]]; then
        error "Backup appears to be empty or incomplete"
        return 1
    fi

    success "Backup validation passed"

    # Show backup info
    if [[ -f "$BACKUP_PATH/MANIFEST.txt" ]]; then
        info "Backup manifest:"
        head -20 "$BACKUP_PATH/MANIFEST.txt" | sed 's/^/  /'
    fi
}

# =============================================================================
# PACKAGE REMOVAL
# =============================================================================

detect_packages() {
    section "DETECTING INSTALLED PACKAGES"

    local packages_found=()

    for package in "$DSMIL_PACKAGE" "$TPM2_PACKAGE" "$TOOLS_PACKAGE"; do
        if dpkg -l | grep -q "^ii.*$package"; then
            info "Found package: $package"
            packages_found+=("$package")
        else
            info "Package not installed: $package"
        fi
    done

    if [[ ${#packages_found[@]} -eq 0 ]]; then
        warning "No Dell MIL-SPEC packages found"
        if [[ "$FORCE_MODE" == "false" ]]; then
            error "Nothing to rollback. Use --force to restore manual installation anyway."
            return 1
        fi
    else
        success "Found ${#packages_found[@]} packages to remove"
    fi

    return 0
}

stop_package_services() {
    section "STOPPING PACKAGE SERVICES"

    # Stop any running monitoring
    if pgrep -f "milspec-monitor" > /dev/null; then
        warning "Found running monitoring processes"
        execute "sudo pkill -f 'milspec-monitor'" "Stopped monitoring processes"
    fi

    # Stop DKMS modules
    local modules=("dsmil" "tpm2_accel_early")
    for module in "${modules[@]}"; do
        if lsmod | grep -q "^${module}"; then
            info "Unloading DKMS module: $module"
            execute "sudo modprobe -r $module" "Unloaded $module" || warning "Could not unload $module"
        fi
    done

    success "Package services stopped"
}

remove_packages() {
    section "REMOVING .DEB PACKAGES"

    local packages_to_remove=()

    for package in "$TOOLS_PACKAGE" "$DSMIL_PACKAGE" "$TPM2_PACKAGE"; do
        if dpkg -l | grep -q "^ii.*$package"; then
            packages_to_remove+=("$package")
        fi
    done

    if [[ ${#packages_to_remove[@]} -eq 0 ]]; then
        info "No packages to remove"
        return 0
    fi

    info "Removing packages: ${packages_to_remove[*]}"

    if [[ "$DRY_RUN" == "false" ]]; then
        if sudo DEBIAN_FRONTEND=noninteractive apt-get purge -y "${packages_to_remove[@]}" \
           >> "$ROLLBACK_LOG" 2>&1; then
            success "Packages removed successfully"
        else
            error "Package removal failed"
            return 1
        fi

        # Clean up
        info "Cleaning up package cache..."
        sudo apt-get autoremove -y >> "$ROLLBACK_LOG" 2>&1
        sudo apt-get clean >> "$ROLLBACK_LOG" 2>&1
    else
        info "[DRY RUN] Would remove: ${packages_to_remove[*]}"
    fi

    success "Package removal complete"
}

remove_package_configs() {
    section "REMOVING PACKAGE CONFIGURATIONS"

    local config_paths=(
        "/etc/dell-milspec"
        "/var/log/dell-milspec"
        "/var/run/dell-milspec"
        "/etc/modprobe.d/dell-milspec.conf"
    )

    for path in "${config_paths[@]}"; do
        if [[ -e "$path" ]]; then
            info "Removing: $path"
            execute "sudo rm -rf '$path'" "Removed $path"
        fi
    done

    # Remove dsmil group if it exists
    if getent group dsmil >/dev/null 2>&1; then
        info "Removing dsmil group..."
        execute "sudo groupdel dsmil" "Removed dsmil group" || warning "Could not remove dsmil group"
    fi

    success "Package configurations removed"
}

# =============================================================================
# RESTORE MANUAL INSTALLATION
# =============================================================================

restore_kernel_modules() {
    section "RESTORING KERNEL MODULES"

    if [[ ! -d "$BACKUP_PATH/modules" ]]; then
        warning "No modules found in backup"
        return 0
    fi

    local modules_restored=0

    # Restore DSMIL module
    local dsmil_module=$(find "$BACKUP_PATH/modules" -name "dsmil-72dev.ko" 2>/dev/null | head -1)
    if [[ -n "$dsmil_module" ]]; then
        local kernel_version=$(uname -r)
        local target_dir="/lib/modules/${kernel_version}/extra"

        info "Restoring DSMIL module to $target_dir"
        execute "sudo mkdir -p '$target_dir'" "Created module directory"
        execute "sudo cp '$dsmil_module' '$target_dir/'" "Restored dsmil-72dev.ko"
        ((modules_restored++))
    fi

    # Restore TPM2 module
    local tpm2_module=$(find "$BACKUP_PATH/modules" -name "tpm2_accel_early.ko" 2>/dev/null | head -1)
    if [[ -n "$tpm2_module" ]]; then
        local kernel_version=$(uname -r)
        local target_dir="/lib/modules/${kernel_version}/kernel/drivers/tpm"

        info "Restoring TPM2 module to $target_dir"
        execute "sudo mkdir -p '$target_dir'" "Created module directory"
        execute "sudo cp '$tpm2_module' '$target_dir/'" "Restored tpm2_accel_early.ko"
        ((modules_restored++))
    fi

    if [[ $modules_restored -gt 0 ]]; then
        info "Updating module dependencies..."
        execute "sudo depmod -a" "Updated module dependencies"
        success "Restored $modules_restored kernel modules"
    else
        warning "No kernel modules to restore"
    fi
}

restore_monitoring_system() {
    section "RESTORING MONITORING SYSTEM"

    if [[ ! -d "$BACKUP_PATH/monitoring" ]]; then
        warning "No monitoring system found in backup"
        return 0
    fi

    local source_dir=$(find "$BACKUP_PATH/monitoring" -type d -name "dsmil" 2>/dev/null | head -1)

    if [[ -z "$source_dir" ]] || [[ ! -d "$source_dir" ]]; then
        warning "Monitoring directory not found in backup"
        return 0
    fi

    info "Restoring monitoring system to $MANUAL_INSTALL_DIR"

    execute "sudo mkdir -p '$MANUAL_INSTALL_DIR'" "Created installation directory"
    execute "sudo cp -rp '$source_dir'/* '$MANUAL_INSTALL_DIR/'" "Restored monitoring files"

    # Set permissions
    execute "sudo chmod -R 755 '$MANUAL_INSTALL_DIR'" "Set directory permissions"

    if [[ -d "$MANUAL_INSTALL_DIR/monitoring" ]]; then
        execute "sudo chmod +x '$MANUAL_INSTALL_DIR'/monitoring/*.sh" "Set script permissions" || true
        execute "sudo chmod +x '$MANUAL_INSTALL_DIR'/monitoring/*.py" "Set Python permissions" || true
    fi

    success "Monitoring system restored"
}

restore_configurations() {
    section "RESTORING CONFIGURATIONS"

    if [[ ! -d "$BACKUP_PATH/configs" ]]; then
        warning "No configurations found in backup"
        return 0
    fi

    local configs_restored=0

    # Restore each configuration file
    for config in "$BACKUP_PATH"/configs/*; do
        if [[ -f "$config" ]]; then
            local config_name=$(basename "$config")
            local target_path=""

            # Determine target path based on filename
            if [[ "$config_name" == *"modprobe.d"* ]] || [[ "$config_name" == "dsmil-72dev.conf" ]]; then
                target_path="/etc/modprobe.d/$config_name"
            elif [[ "$config_name" == *"modules-load.d"* ]] || [[ "$config_name" == "tpm2-acceleration.conf" ]]; then
                target_path="/etc/modules-load.d/$config_name"
            elif [[ "$config_name" == "tpm2-acceleration.conf" ]]; then
                target_path="/etc/modprobe.d/$config_name"
            else
                warning "Unknown config file: $config_name"
                continue
            fi

            info "Restoring: $config_name -> $target_path"
            execute "sudo cp '$config' '$target_path'" "Restored $config_name"
            ((configs_restored++))
        fi
    done

    if [[ $configs_restored -gt 0 ]]; then
        success "Restored $configs_restored configuration files"
    else
        warning "No configuration files restored"
    fi
}

restore_systemd_service() {
    section "RESTORING SYSTEMD SERVICE"

    if [[ ! -d "$BACKUP_PATH/services" ]]; then
        warning "No services found in backup"
        return 0
    fi

    local service_file=$(find "$BACKUP_PATH/services" -name "dsmil-monitor.service" 2>/dev/null | head -1)

    if [[ -n "$service_file" ]] && [[ -f "$service_file" ]]; then
        info "Restoring systemd service..."
        execute "sudo cp '$service_file' '$MANUAL_SERVICE'" "Restored dsmil-monitor.service"
        execute "sudo systemctl daemon-reload" "Reloaded systemd"

        if confirm "Enable dsmil-monitor service?" "n"; then
            execute "sudo systemctl enable dsmil-monitor" "Enabled service"
        fi

        success "Systemd service restored"
    else
        warning "No systemd service found in backup"
    fi
}

# =============================================================================
# VALIDATION
# =============================================================================

validate_manual_installation() {
    section "VALIDATING MANUAL INSTALLATION"

    local errors=0

    # Check kernel modules exist
    local kernel_version=$(uname -r)
    local dsmil_module="/lib/modules/${kernel_version}/extra/dsmil-72dev.ko"

    if [[ -f "$dsmil_module" ]]; then
        success "DSMIL module restored: $dsmil_module"
    else
        error "DSMIL module not found: $dsmil_module"
        ((errors++))
    fi

    # Check monitoring directory
    if [[ -d "$MANUAL_INSTALL_DIR" ]]; then
        success "Monitoring directory restored: $MANUAL_INSTALL_DIR"

        # Check for key files
        local key_files=("monitoring" "config")
        for dir in "${key_files[@]}"; do
            if [[ -d "$MANUAL_INSTALL_DIR/$dir" ]]; then
                success "Found subdirectory: $dir"
            else
                warning "Missing subdirectory: $dir"
            fi
        done
    else
        error "Monitoring directory not found: $MANUAL_INSTALL_DIR"
        ((errors++))
    fi

    # Check no packages remain
    for package in "$DSMIL_PACKAGE" "$TPM2_PACKAGE" "$TOOLS_PACKAGE"; do
        if dpkg -l | grep -q "^ii.*$package"; then
            error "Package still installed: $package"
            ((errors++))
        else
            success "Package removed: $package"
        fi
    done

    if [[ $errors -eq 0 ]]; then
        success "Manual installation validation passed"
        return 0
    else
        error "Manual installation validation failed with $errors errors"
        return 1
    fi
}

load_manual_modules() {
    section "LOADING MANUAL MODULES"

    if ! confirm "Load manual kernel modules now?" "y"; then
        info "Skipping module loading"
        return 0
    fi

    local modules=("dsmil-72dev" "tpm2_accel_early")

    for module in "${modules[@]}"; do
        info "Loading module: $module"

        if [[ "$DRY_RUN" == "false" ]]; then
            if sudo modprobe "$module" 2>> "$ROLLBACK_LOG"; then
                success "Module loaded: $module"

                # Verify
                if lsmod | grep -q "^${module}"; then
                    success "Module verified in kernel"
                else
                    warning "Module not found in lsmod"
                fi
            else
                warning "Failed to load module: $module"
            fi
        else
            info "[DRY RUN] Would load: $module"
        fi
    done

    success "Module loading complete"
}

# =============================================================================
# REPORTING
# =============================================================================

generate_rollback_report() {
    section "GENERATING ROLLBACK REPORT"

    local report="${BACKUP_PATH}/ROLLBACK_REPORT.txt"

    if [[ "$DRY_RUN" == "true" ]]; then
        report="/tmp/rollback_report_dryrun.txt"
    fi

    cat > "$report" << EOF
Dell MIL-SPEC Platform - Rollback from .deb Packages
=====================================================

Rollback Date: $(date)
Rollback Script Version: $SCRIPT_VERSION
Dry Run: $DRY_RUN

System Information:
-------------------
Hostname: $(hostname)
Kernel: $(uname -r)
OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)

Backup Used:
------------
Backup Location: $BACKUP_PATH
Backup Manifest: $BACKUP_PATH/MANIFEST.txt

Rollback Summary:
-----------------
Packages Removed: $TOOLS_PACKAGE, $DSMIL_PACKAGE, $TPM2_PACKAGE
Manual Installation Restored: $MANUAL_INSTALL_DIR

Restored Components:
--------------------
EOF

    if [[ -f "/lib/modules/$(uname -r)/extra/dsmil-72dev.ko" ]]; then
        echo "- DSMIL kernel module" >> "$report"
    fi

    if [[ -f "/lib/modules/$(uname -r)/kernel/drivers/tpm/tpm2_accel_early.ko" ]]; then
        echo "- TPM2 kernel module" >> "$report"
    fi

    if [[ -d "$MANUAL_INSTALL_DIR" ]]; then
        echo "- Monitoring system: $MANUAL_INSTALL_DIR" >> "$report"
    fi

    if [[ -f "$MANUAL_SERVICE" ]]; then
        echo "- Systemd service: $MANUAL_SERVICE" >> "$report"
    fi

    cat >> "$report" << EOF

Next Steps:
-----------
1. Load kernel modules (if not already loaded):
   sudo modprobe dsmil-72dev
   sudo modprobe tpm2_accel_early

2. Verify modules loaded:
   lsmod | grep -E "dsmil|tpm2_accel"

3. Check device files:
   ls -la /dev/dsmil* /dev/tpm2*

4. Test monitoring system:
   cd $MANUAL_INSTALL_DIR/monitoring
   python3 dsmil_comprehensive_monitor.py --help

5. Start systemd service (if desired):
   sudo systemctl start dsmil-monitor
   sudo systemctl status dsmil-monitor

Manual Installation Location:
------------------------------
Install Directory: $MANUAL_INSTALL_DIR
Service File: $MANUAL_SERVICE
Configuration: $MANUAL_INSTALL_DIR/config/

Rollback Log:
-------------
Full log available at: $ROLLBACK_LOG

EOF

    success "Rollback report created: $report"

    if [[ "$VERBOSE" == "true" ]]; then
        cat "$report"
    fi
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

show_usage() {
    cat << EOF
Dell MIL-SPEC Platform - Migration Rollback
Version: $SCRIPT_VERSION

USAGE:
    sudo $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -n, --dry-run           Dry run mode (no actual changes)
    -f, --force             Force rollback (bypass checks)
    -y, --yes               Auto-confirm all prompts
    -b, --backup PATH       Specify backup directory path
    -v, --verbose           Verbose output

DESCRIPTION:
    Rolls back from .deb packages to manual installation.

    This script will:
    1. Detect installed .deb packages
    2. Locate manual installation backup
    3. Remove .deb packages cleanly
    4. Restore manual installation from backup
    5. Restore configurations
    6. Validate manual installation
    7. Generate rollback report

SAFETY:
    - Validates backup before rollback
    - Supports dry-run mode
    - Detailed logging
    - Can re-migrate later if needed

EXAMPLES:
    sudo $0 --dry-run           # Preview rollback
    sudo $0 --verbose           # Detailed output
    sudo $0 --backup /path      # Use specific backup

REQUIREMENTS:
    - Root privileges (sudo)
    - Valid backup directory
    - .deb packages installed (or --force)

EOF
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -n|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -f|--force)
                FORCE_MODE=true
                shift
                ;;
            -y|--yes)
                AUTO_CONFIRM=true
                shift
                ;;
            -b|--backup)
                BACKUP_PATH="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            *)
                error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

main() {
    # Parse arguments
    parse_arguments "$@"

    # Initialize
    section "DELL MIL-SPEC PLATFORM - ROLLBACK FROM .DEB PACKAGES v$SCRIPT_VERSION"

    if [[ "$DRY_RUN" == "true" ]]; then
        warning "DRY RUN MODE - No actual changes will be made"
    fi

    # Check root
    check_root

    # Initialize logging
    init_logging

    # Detect packages
    if ! detect_packages; then
        if [[ "$FORCE_MODE" == "false" ]]; then
            exit 1
        fi
    fi

    # Find backup
    if ! find_backup; then
        critical "Cannot proceed without valid backup"
        exit 1
    fi

    # Show summary and confirm
    echo
    warning "This will remove .deb packages and restore manual installation"
    info "Backup to restore: $BACKUP_PATH"
    echo

    if ! confirm "Proceed with rollback?" "n"; then
        info "Rollback cancelled by user"
        exit 0
    fi

    # Execute rollback steps
    stop_package_services || { critical "Service stop failed"; exit 1; }
    remove_packages || { critical "Package removal failed"; exit 1; }
    remove_package_configs || { warning "Config removal had issues"; }
    restore_kernel_modules || { critical "Module restoration failed"; exit 1; }
    restore_monitoring_system || { critical "Monitoring restoration failed"; exit 1; }
    restore_configurations || { warning "Configuration restoration had issues"; }
    restore_systemd_service || { warning "Service restoration had issues"; }
    validate_manual_installation || { warning "Validation had issues"; }
    load_manual_modules || { warning "Module loading had issues"; }
    generate_rollback_report

    # Success
    section "ROLLBACK COMPLETE"

    success "Rollback completed successfully!"
    echo
    info "Manual installation restored from: $BACKUP_PATH"
    info "Rollback log: $ROLLBACK_LOG"
    echo
    info "Next steps:"
    echo "  1. Verify modules: lsmod | grep -E 'dsmil|tpm2'"
    echo "  2. Check devices: ls -la /dev/dsmil* /dev/tpm2*"
    echo "  3. Test monitoring: cd $MANUAL_INSTALL_DIR"
    echo

    if [[ "$DRY_RUN" == "false" ]]; then
        warning "A reboot may be required for complete restoration"
    fi

    log "Rollback completed successfully"
}

# Run main
main "$@"

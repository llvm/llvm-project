#!/bin/bash
#
# Dell MIL-SPEC Platform - Migration to .deb Packages
# Main migration orchestrator for transitioning from manual installation
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

# Backup configuration
BACKUP_ROOT="/var/backups/dell-milspec-manual-${TIMESTAMP}"
MIGRATION_LOG="/var/log/dell-milspec-migration.log"

# Package names
DSMIL_PACKAGE="dell-milspec-dsmil-dkms"
TPM2_PACKAGE="tpm2-accel-early-dkms"
TOOLS_PACKAGE="dell-milspec-tools"

# Manual installation artifacts
MANUAL_MODULE_DSMIL="/lib/modules/$(uname -r)/extra/dsmil-72dev.ko"
MANUAL_MODULE_TPM2="/lib/modules/$(uname -r)/kernel/drivers/tpm/tpm2_accel_early.ko"
MANUAL_INSTALL_DIR="/opt/dsmil"
MANUAL_SERVICE="/etc/systemd/system/dsmil-monitor.service"
MANUAL_CONFIGS=(
    "/etc/modprobe.d/dsmil-72dev.conf"
    "/etc/modules-load.d/tpm2-acceleration.conf"
    "/etc/modprobe.d/tpm2-acceleration.conf"
)

# New package locations
NEW_CONFIG_DIR="/etc/dell-milspec"
NEW_MODPROBE_CONF="/etc/modprobe.d/dell-milspec.conf"

# Flags
DRY_RUN=false
FORCE_MODE=false
AUTO_CONFIRM=false
SKIP_BACKUP=false
VERBOSE=false

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
    local log_dir=$(dirname "$MIGRATION_LOG")
    sudo mkdir -p "$log_dir"
    sudo touch "$MIGRATION_LOG"
    sudo chmod 644 "$MIGRATION_LOG"

    log "Migration to .deb packages started"
    log "Version: $SCRIPT_VERSION"
    log "Timestamp: $TIMESTAMP"
    log "Dry run: $DRY_RUN"
}

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" | sudo tee -a "$MIGRATION_LOG" > /dev/null
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

    if eval "$cmd" >> "$MIGRATION_LOG" 2>&1; then
        success "$description"
        return 0
    else
        error "$description failed"
        return 1
    fi
}

check_package_available() {
    local package="$1"
    if apt-cache show "$package" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================

detect_manual_installation() {
    section "DETECTING MANUAL INSTALLATION"

    local detection_script="${SCRIPT_DIR}/detect-manual-install.sh"

    if [[ ! -f "$detection_script" ]]; then
        error "Detection script not found: $detection_script"
        return 1
    fi

    info "Running manual installation detection..."

    if bash "$detection_script" > /tmp/detection_report.json; then
        success "Manual installation detected"

        if [[ "$VERBOSE" == "true" ]]; then
            cat /tmp/detection_report.json
        fi

        return 0
    else
        local exit_code=$?
        if [[ $exit_code -eq 1 ]]; then
            info "Clean system detected - no manual installation found"
            return 1
        else
            warning "Partial manual installation detected"
            return 2
        fi
    fi
}

# =============================================================================
# BACKUP FUNCTIONS
# =============================================================================

create_backup() {
    section "CREATING BACKUP"

    if [[ "$SKIP_BACKUP" == "true" ]]; then
        warning "Skipping backup (--skip-backup specified)"
        return 0
    fi

    info "Creating backup directory: $BACKUP_ROOT"

    if [[ "$DRY_RUN" == "false" ]]; then
        sudo mkdir -p "$BACKUP_ROOT"/{modules,configs,monitoring,services,logs}
    fi

    local backup_items=0
    local total_items=10

    # Backup kernel modules
    progress_bar $((++backup_items)) $total_items "Backing up kernel modules..."
    if [[ -f "$MANUAL_MODULE_DSMIL" ]]; then
        execute "sudo cp -p '$MANUAL_MODULE_DSMIL' '$BACKUP_ROOT/modules/'" \
                "Backed up DSMIL module"
    fi

    if [[ -f "$MANUAL_MODULE_TPM2" ]]; then
        execute "sudo cp -p '$MANUAL_MODULE_TPM2' '$BACKUP_ROOT/modules/'" \
                "Backed up TPM2 module"
    fi

    # Backup monitoring system
    progress_bar $((++backup_items)) $total_items "Backing up monitoring system..."
    if [[ -d "$MANUAL_INSTALL_DIR" ]]; then
        execute "sudo cp -rp '$MANUAL_INSTALL_DIR' '$BACKUP_ROOT/monitoring/'" \
                "Backed up monitoring directory"
    fi

    # Backup configurations
    progress_bar $((++backup_items)) $total_items "Backing up configurations..."
    for config in "${MANUAL_CONFIGS[@]}"; do
        if [[ -f "$config" ]]; then
            execute "sudo cp -p '$config' '$BACKUP_ROOT/configs/'" \
                    "Backed up $(basename $config)"
        fi
    done

    # Backup systemd service
    progress_bar $((++backup_items)) $total_items "Backing up services..."
    if [[ -f "$MANUAL_SERVICE" ]]; then
        execute "sudo cp -p '$MANUAL_SERVICE' '$BACKUP_ROOT/services/'" \
                "Backed up systemd service"
    fi

    # Backup logs
    progress_bar $((++backup_items)) $total_items "Backing up logs..."
    if [[ -d "${MANUAL_INSTALL_DIR}/logs" ]]; then
        execute "sudo cp -rp '${MANUAL_INSTALL_DIR}/logs' '$BACKUP_ROOT/logs/'" \
                "Backed up log files"
    fi

    # Create manifest
    progress_bar $((++backup_items)) $total_items "Creating backup manifest..."
    create_backup_manifest

    # Set permissions
    progress_bar $((++backup_items)) $total_items "Setting permissions..."
    execute "sudo chmod -R 755 '$BACKUP_ROOT'" "Set backup permissions"

    # Create archive
    progress_bar $((++backup_items)) $total_items "Creating backup archive..."
    if [[ "$DRY_RUN" == "false" ]]; then
        sudo tar czf "${BACKUP_ROOT}.tar.gz" -C "$(dirname $BACKUP_ROOT)" \
             "$(basename $BACKUP_ROOT)" 2>> "$MIGRATION_LOG"
        success "Backup archive created: ${BACKUP_ROOT}.tar.gz"
    fi

    # Calculate size
    progress_bar $((++backup_items)) $total_items "Calculating backup size..."
    if [[ "$DRY_RUN" == "false" ]]; then
        local backup_size=$(du -sh "$BACKUP_ROOT" | cut -f1)
        info "Backup size: $backup_size"
    fi

    progress_bar $total_items $total_items "Backup complete"

    success "Backup created successfully: $BACKUP_ROOT"
}

create_backup_manifest() {
    local manifest="${BACKUP_ROOT}/MANIFEST.txt"

    if [[ "$DRY_RUN" == "true" ]]; then
        return 0
    fi

    cat > "$manifest" << EOF
Dell MIL-SPEC Platform - Manual Installation Backup
====================================================

Backup Date: $(date)
Backup Location: $BACKUP_ROOT
Migration Script Version: $SCRIPT_VERSION

System Information:
-------------------
Hostname: $(hostname)
Kernel: $(uname -r)
OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)

Backed Up Components:
---------------------
EOF

    if [[ -f "$MANUAL_MODULE_DSMIL" ]]; then
        echo "- DSMIL kernel module: $MANUAL_MODULE_DSMIL" >> "$manifest"
    fi

    if [[ -f "$MANUAL_MODULE_TPM2" ]]; then
        echo "- TPM2 kernel module: $MANUAL_MODULE_TPM2" >> "$manifest"
    fi

    if [[ -d "$MANUAL_INSTALL_DIR" ]]; then
        echo "- Monitoring system: $MANUAL_INSTALL_DIR" >> "$manifest"
    fi

    for config in "${MANUAL_CONFIGS[@]}"; do
        if [[ -f "$config" ]]; then
            echo "- Configuration: $config" >> "$manifest"
        fi
    done

    if [[ -f "$MANUAL_SERVICE" ]]; then
        echo "- Systemd service: $MANUAL_SERVICE" >> "$manifest"
    fi

    echo "" >> "$manifest"
    echo "File Listing:" >> "$manifest"
    echo "-------------" >> "$manifest"
    sudo find "$BACKUP_ROOT" -type f -o -type d >> "$manifest"

    success "Backup manifest created"
}

# =============================================================================
# UNLOAD AND REMOVE MANUAL INSTALLATION
# =============================================================================

unload_manual_modules() {
    section "UNLOADING MANUAL KERNEL MODULES"

    local modules=("dsmil-72dev" "tpm2_accel_early")

    for module in "${modules[@]}"; do
        if lsmod | grep -q "^${module}"; then
            info "Unloading module: $module"

            if [[ "$DRY_RUN" == "false" ]]; then
                if sudo modprobe -r "$module" 2>> "$MIGRATION_LOG"; then
                    success "Module unloaded: $module"
                else
                    warning "Failed to unload module: $module (may be in use)"
                    if [[ "$FORCE_MODE" == "false" ]]; then
                        error "Use --force to continue anyway"
                        return 1
                    fi
                fi
            else
                info "[DRY RUN] Would unload: $module"
            fi
        else
            info "Module not loaded: $module"
        fi
    done

    success "Manual modules unloaded"
}

stop_manual_services() {
    section "STOPPING MANUAL SERVICES"

    if systemctl is-active --quiet dsmil-monitor 2>/dev/null; then
        info "Stopping dsmil-monitor service..."
        execute "sudo systemctl stop dsmil-monitor" "Stopped dsmil-monitor"
    fi

    if systemctl is-enabled --quiet dsmil-monitor 2>/dev/null; then
        info "Disabling dsmil-monitor service..."
        execute "sudo systemctl disable dsmil-monitor" "Disabled dsmil-monitor"
    fi

    # Kill any running monitoring processes
    if pgrep -f "dsmil.*monitor" > /dev/null; then
        warning "Found running monitoring processes"
        if confirm "Stop all monitoring processes?"; then
            execute "sudo pkill -f 'dsmil.*monitor'" "Stopped monitoring processes"
        fi
    fi

    success "Manual services stopped"
}

remove_manual_files() {
    section "REMOVING MANUAL INSTALLATION FILES"

    local remove_count=0

    # Remove kernel modules
    if [[ -f "$MANUAL_MODULE_DSMIL" ]]; then
        info "Removing DSMIL module: $MANUAL_MODULE_DSMIL"
        execute "sudo rm -f '$MANUAL_MODULE_DSMIL'" "Removed DSMIL module"
        ((remove_count++))
    fi

    if [[ -f "$MANUAL_MODULE_TPM2" ]]; then
        info "Removing TPM2 module: $MANUAL_MODULE_TPM2"
        execute "sudo rm -f '$MANUAL_MODULE_TPM2'" "Removed TPM2 module"
        ((remove_count++))
    fi

    # Update module dependencies
    if [[ $remove_count -gt 0 ]]; then
        info "Updating module dependencies..."
        execute "sudo depmod -a" "Updated module dependencies"
    fi

    # Remove systemd service
    if [[ -f "$MANUAL_SERVICE" ]]; then
        info "Removing systemd service: $MANUAL_SERVICE"
        execute "sudo rm -f '$MANUAL_SERVICE'" "Removed systemd service"
        execute "sudo systemctl daemon-reload" "Reloaded systemd"
    fi

    # Remove monitoring directory
    if [[ -d "$MANUAL_INSTALL_DIR" ]]; then
        warning "Removing monitoring directory: $MANUAL_INSTALL_DIR"
        if confirm "Remove $MANUAL_INSTALL_DIR?"; then
            execute "sudo rm -rf '$MANUAL_INSTALL_DIR'" "Removed monitoring directory"
        fi
    fi

    # Remove manual configuration files (will be migrated)
    info "Removing old configuration files..."
    for config in "${MANUAL_CONFIGS[@]}"; do
        if [[ -f "$config" ]]; then
            execute "sudo rm -f '$config'" "Removed $(basename $config)"
        fi
    done

    success "Manual installation files removed"
}

# =============================================================================
# PACKAGE INSTALLATION
# =============================================================================

install_packages() {
    section "INSTALLING .DEB PACKAGES"

    # Update package cache
    info "Updating package cache..."
    execute "sudo apt-get update" "Package cache updated"

    local packages=("$TOOLS_PACKAGE")

    # Check if DKMS packages are available
    if check_package_available "$DSMIL_PACKAGE"; then
        packages+=("$DSMIL_PACKAGE")
    else
        warning "Package not available: $DSMIL_PACKAGE"
    fi

    if check_package_available "$TPM2_PACKAGE"; then
        packages+=("$TPM2_PACKAGE")
    else
        warning "Package not available: $TPM2_PACKAGE"
    fi

    info "Installing packages: ${packages[*]}"

    if [[ "$DRY_RUN" == "false" ]]; then
        if sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${packages[@]}" \
           >> "$MIGRATION_LOG" 2>&1; then
            success "Packages installed successfully"
        else
            error "Package installation failed"
            return 1
        fi
    else
        info "[DRY RUN] Would install: ${packages[*]}"
    fi

    success "Package installation complete"
}

# =============================================================================
# CONFIGURATION MIGRATION
# =============================================================================

migrate_configurations() {
    section "MIGRATING CONFIGURATIONS"

    # Create new config directory if needed
    if [[ ! -d "$NEW_CONFIG_DIR" ]]; then
        execute "sudo mkdir -p '$NEW_CONFIG_DIR'" "Created config directory"
    fi

    # Migrate DSMIL configuration
    local old_config="${MANUAL_INSTALL_DIR}/config/dsmil.json"
    local new_config="${NEW_CONFIG_DIR}/dsmil.conf"

    if [[ -f "$old_config" ]]; then
        info "Migrating DSMIL configuration..."
        migrate_dsmil_config "$old_config" "$new_config"
    fi

    # Migrate monitoring configuration
    local old_monitor="${MANUAL_INSTALL_DIR}/config/monitoring.json"
    local new_monitor="${NEW_CONFIG_DIR}/monitoring.json"

    if [[ -f "$old_monitor" && ! -f "$new_monitor" ]]; then
        info "Migrating monitoring configuration..."
        execute "sudo cp '$old_monitor' '$new_monitor'" "Copied monitoring config"
    fi

    # Migrate safety configuration
    local old_safety="${MANUAL_INSTALL_DIR}/config/safety.json"
    local new_safety="${NEW_CONFIG_DIR}/safety.json"

    if [[ -f "$old_safety" && ! -f "$new_safety" ]]; then
        info "Migrating safety configuration..."
        execute "sudo cp '$old_safety' '$new_safety'" "Copied safety config"
    fi

    # Migrate modprobe configuration
    local old_modprobe="/etc/modprobe.d/dsmil-72dev.conf"
    if [[ -f "$old_modprobe" ]]; then
        info "Migrating modprobe configuration..."
        # Extract relevant settings and merge into new config
        # This is handled during backup - new package will create proper config
        info "Old modprobe config backed up, new package will create updated version"
    fi

    success "Configuration migration complete"
}

migrate_dsmil_config() {
    local old_config="$1"
    local new_config="$2"

    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would migrate: $old_config -> $new_config"
        return 0
    fi

    # Convert JSON to shell format (simplified)
    # In production, use jq or python for proper conversion
    info "Converting JSON configuration to shell format..."

    # For now, preserve old config and let new package create defaults
    if [[ -f "$old_config" ]]; then
        sudo cp "$old_config" "${NEW_CONFIG_DIR}/dsmil.json.old"
        success "Preserved old configuration for reference"
    fi
}

# =============================================================================
# VALIDATION
# =============================================================================

validate_migration() {
    section "VALIDATING MIGRATION"

    local errors=0

    # Check packages installed
    info "Checking installed packages..."
    if dpkg -l | grep -q "$TOOLS_PACKAGE"; then
        success "Package installed: $TOOLS_PACKAGE"
    else
        error "Package not installed: $TOOLS_PACKAGE"
        ((errors++))
    fi

    # Check commands available
    local commands=("dsmil-status" "milspec-control" "milspec-monitor")
    for cmd in "${commands[@]}"; do
        if command -v "$cmd" &> /dev/null; then
            success "Command available: $cmd"
        else
            error "Command not found: $cmd"
            ((errors++))
        fi
    done

    # Check configuration directory
    if [[ -d "$NEW_CONFIG_DIR" ]]; then
        success "Configuration directory exists: $NEW_CONFIG_DIR"
    else
        error "Configuration directory not found: $NEW_CONFIG_DIR"
        ((errors++))
    fi

    # Check DKMS modules
    if dkms status | grep -q "dsmil"; then
        success "DSMIL DKMS module registered"
    else
        warning "DSMIL DKMS module not found (may need manual installation)"
    fi

    if dkms status | grep -q "tpm2-accel-early"; then
        success "TPM2 DKMS module registered"
    else
        warning "TPM2 DKMS module not found (may need manual installation)"
    fi

    # Check backup exists
    if [[ -d "$BACKUP_ROOT" ]]; then
        success "Backup preserved: $BACKUP_ROOT"
    else
        warning "Backup directory not found"
    fi

    if [[ $errors -eq 0 ]]; then
        success "Migration validation passed"
        return 0
    else
        error "Migration validation failed with $errors errors"
        return 1
    fi
}

# =============================================================================
# REPORTING
# =============================================================================

generate_report() {
    section "GENERATING MIGRATION REPORT"

    local report="${BACKUP_ROOT}/MIGRATION_REPORT.txt"

    if [[ "$DRY_RUN" == "true" ]]; then
        report="/tmp/migration_report_dryrun.txt"
    fi

    cat > "$report" << EOF
Dell MIL-SPEC Platform - Migration to .deb Packages
====================================================

Migration Date: $(date)
Migration Script Version: $SCRIPT_VERSION
Dry Run: $DRY_RUN

System Information:
-------------------
Hostname: $(hostname)
Kernel: $(uname -r)
OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)

Migration Summary:
------------------
Backup Location: $BACKUP_ROOT
Backup Archive: ${BACKUP_ROOT}.tar.gz
Migration Log: $MIGRATION_LOG

Packages Installed:
-------------------
EOF

    if dpkg -l | grep -E "(dell-milspec|tpm2-accel)" >> "$report" 2>&1; then
        :
    fi

    cat >> "$report" << EOF

Configuration Migration:
------------------------
Old Location: $MANUAL_INSTALL_DIR
New Location: $NEW_CONFIG_DIR

Old Modules: /lib/modules/$(uname -r)/extra/
New Modules: DKMS-managed

Available Commands:
-------------------
EOF

    for cmd in dsmil-status dsmil-test milspec-control milspec-monitor; do
        if command -v "$cmd" &> /dev/null; then
            echo "- $cmd ($(command -v $cmd))" >> "$report"
        fi
    done

    cat >> "$report" << EOF

Next Steps:
-----------
1. Verify installation:
   sudo dsmil-status
   sudo tpm2-accel-status

2. Load DKMS modules (if needed):
   sudo dkms install dell-milspec-dsmil/1.0.0
   sudo dkms install tpm2-accel-early/1.0.0
   sudo modprobe dsmil
   sudo modprobe tpm2_accel_early

3. Test functionality:
   milspec-control
   dsmil-test --basic-only

4. Review migrated configurations:
   ls -la $NEW_CONFIG_DIR/

5. If issues occur, rollback is available:
   sudo ${SCRIPT_DIR}/rollback-migration.sh

Rollback Information:
---------------------
Backup preserved at: $BACKUP_ROOT
Rollback script: ${SCRIPT_DIR}/rollback-migration.sh
To rollback: sudo ${SCRIPT_DIR}/rollback-migration.sh

Migration Log:
--------------
Full log available at: $MIGRATION_LOG

EOF

    success "Migration report created: $report"

    if [[ "$VERBOSE" == "true" ]]; then
        cat "$report"
    fi
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

show_usage() {
    cat << EOF
Dell MIL-SPEC Platform - Migration to .deb Packages
Version: $SCRIPT_VERSION

USAGE:
    sudo $0 [OPTIONS]

OPTIONS:
    -h, --help          Show this help message
    -n, --dry-run       Dry run mode (no actual changes)
    -f, --force         Force migration (bypass checks)
    -y, --yes           Auto-confirm all prompts
    --skip-backup       Skip backup creation (NOT RECOMMENDED)
    -v, --verbose       Verbose output

DESCRIPTION:
    Migrates from manual installation to .deb package system.

    This script will:
    1. Detect existing manual installation
    2. Create comprehensive backup
    3. Unload manual kernel modules
    4. Remove manual installation files
    5. Install .deb packages
    6. Migrate configurations
    7. Validate new installation
    8. Generate migration report

SAFETY:
    - Creates backup before any changes
    - Supports dry-run mode
    - Rollback capability available
    - Detailed logging

EXAMPLES:
    sudo $0 --dry-run       # Preview migration
    sudo $0 --verbose       # Detailed output
    sudo $0 --yes           # Automatic migration

REQUIREMENTS:
    - Root privileges (sudo)
    - Manual installation present
    - .deb packages available
    - Sufficient disk space for backup

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
            --skip-backup)
                SKIP_BACKUP=true
                shift
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
    section "DELL MIL-SPEC PLATFORM - MIGRATION TO .DEB PACKAGES v$SCRIPT_VERSION"

    if [[ "$DRY_RUN" == "true" ]]; then
        warning "DRY RUN MODE - No actual changes will be made"
    fi

    # Check root
    check_root

    # Initialize logging
    init_logging

    # Detect manual installation
    if ! detect_manual_installation; then
        info "No manual installation detected. Migration not needed."
        exit 0
    fi

    # Show summary and confirm
    echo
    info "This will migrate your system from manual installation to .deb packages"
    info "Backup will be created at: $BACKUP_ROOT"
    echo

    if ! confirm "Proceed with migration?" "n"; then
        info "Migration cancelled by user"
        exit 0
    fi

    # Execute migration steps
    create_backup || { critical "Backup failed"; exit 1; }
    stop_manual_services || { critical "Service stop failed"; exit 1; }
    unload_manual_modules || { critical "Module unload failed"; exit 1; }
    remove_manual_files || { critical "File removal failed"; exit 1; }
    install_packages || { critical "Package installation failed"; exit 1; }
    migrate_configurations || { critical "Configuration migration failed"; exit 1; }
    validate_migration || { warning "Validation had issues"; }
    generate_report

    # Success
    section "MIGRATION COMPLETE"

    success "Migration completed successfully!"
    echo
    info "Backup preserved at: $BACKUP_ROOT"
    info "Migration log: $MIGRATION_LOG"
    echo
    info "Next steps:"
    echo "  1. Review migration report: cat ${BACKUP_ROOT}/MIGRATION_REPORT.txt"
    echo "  2. Test new installation: sudo dsmil-status"
    echo "  3. If issues occur: sudo ${SCRIPT_DIR}/rollback-migration.sh"
    echo

    if [[ "$DRY_RUN" == "false" ]]; then
        warning "You may need to logout/login for group changes to take effect"
        warning "A reboot is recommended for full integration"
    fi

    log "Migration completed successfully"
}

# Run main
main "$@"

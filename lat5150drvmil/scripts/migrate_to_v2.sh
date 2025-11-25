#!/bin/bash
#
# DSMIL Platform Migration Script: v1.x → v2.0
# ==============================================
#
# This script automates the migration from legacy DSMIL v1.x components
# to the new v2.0 unified architecture.
#
# Usage: ./migrate_to_v2.sh [--dry-run] [--no-backup]
#
# Options:
#   --dry-run     Show what would be changed without making changes
#   --no-backup   Skip creating backup (not recommended)
#   --help        Show this help message
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DRY_RUN=false
CREATE_BACKUP=true
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_DIR="${SCRIPT_DIR}/_migration_backup_$(date +%Y%m%d_%H%M%S)"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-backup)
            CREATE_BACKUP=false
            shift
            ;;
        --help)
            grep '^#' "$0" | sed 's/^# //; s/^#//'
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Helper functions
print_header() {
    echo -e "\n${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC} $1"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}\n"
}

print_step() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Banner
clear
echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║              DSMIL Platform Migration Tool                       ║
║              Version 1.x → Version 2.0                           ║
║                                                                   ║
║  This will update your codebase to use the new unified APIs     ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

if [ "$DRY_RUN" = true ]; then
    print_warning "DRY RUN MODE - No changes will be made"
fi

# Check if we're in the right directory
if [ ! -f "dsmil.py" ] || [ ! -d "02-ai-engine" ]; then
    print_error "This doesn't appear to be a DSMIL project root directory"
    print_error "Please run this script from the LAT5150DRVMIL directory"
    exit 1
fi

# Create backup
if [ "$CREATE_BACKUP" = true ] && [ "$DRY_RUN" = false ]; then
    print_header "Step 1/6: Creating Backup"
    mkdir -p "$BACKUP_DIR"

    # Backup Python files
    print_step "Backing up Python files..."
    find . -name "*.py" -type f | while read -r file; do
        backup_path="$BACKUP_DIR/$file"
        mkdir -p "$(dirname "$backup_path")"
        cp "$file" "$backup_path"
    done

    # Backup shell scripts
    print_step "Backing up shell scripts..."
    find . -name "*.sh" -type f | while read -r file; do
        backup_path="$BACKUP_DIR/$file"
        mkdir -p "$(dirname "$backup_path")"
        cp "$file" "$backup_path"
    done

    print_step "Backup created at: $BACKUP_DIR"
else
    print_header "Step 1/6: Skipping Backup"
    if [ "$DRY_RUN" = true ]; then
        print_warning "Dry run mode - backup skipped"
    elif [ "$CREATE_BACKUP" = false ]; then
        print_warning "Backup disabled with --no-backup"
    fi
fi

# Update Python imports
print_header "Step 2/6: Updating Python Imports"

if [ "$DRY_RUN" = true ]; then
    print_warning "Would update imports in Python files:"
    grep -rl "from dsmil_device_database import" . --include="*.py" 2>/dev/null || echo "  (none found)"
else
    # Update database imports
    print_step "Updating database imports..."
    find . -name "*.py" -type f -exec sed -i \
        's/from dsmil_device_database import/from dsmil_device_database_extended import/g' {} +

    # Update variable names
    print_step "Updating variable names..."
    find . -name "*.py" -type f -exec sed -i \
        's/\bALL_DEVICES\b/ALL_DEVICES_EXTENDED/g' {} +
    find . -name "*.py" -type f -exec sed -i \
        's/\bget_device(/get_device_extended(/g' {} +
    find . -name "*.py" -type f -exec sed -i \
        's/\bQUARANTINED_DEVICES\b/QUARANTINED_DEVICES_EXTENDED/g' {} +
    find . -name "*.py" -type f -exec sed -i \
        's/\bSAFE_DEVICES\b/SAFE_DEVICES_EXTENDED/g' {} +

    print_step "Python imports updated"
fi

# Update control centre references
print_header "Step 3/6: Updating Control Centre References"

if [ "$DRY_RUN" = true ]; then
    print_warning "Would update control centre calls in:"
    grep -rl "dsmil_subsystem_controller" . --include="*.py" 2>/dev/null || echo "  (none found)"
else
    print_step "Updating control centre module names..."
    find . -name "*.py" -type f -exec sed -i \
        's/dsmil_subsystem_controller/dsmil_control_centre_104/g' {} +
    find . -name "*.py" -type f -exec sed -i \
        's/dsmil_operation_monitor/dsmil_control_centre_104/g' {} +
    find . -name "*.py" -type f -exec sed -i \
        's/dsmil_guided_activation/dsmil_control_centre_104/g' {} +

    print_step "Control centre references updated"
fi

# Update driver references
print_header "Step 4/6: Updating Driver References"

if [ "$DRY_RUN" = true ]; then
    print_warning "Would update driver references in:"
    grep -rl "dsmil-84dev" . --include="*.sh" --include="*.py" 2>/dev/null || echo "  (none found)"
else
    print_step "Updating driver module names..."
    find . \( -name "*.sh" -o -name "*.py" \) -type f -exec sed -i \
        's/dsmil-84dev/dsmil-104dev/g' {} +

    print_step "Driver references updated"
fi

# Update discovery function calls
print_header "Step 5/6: Updating Discovery Function Calls"

if [ "$DRY_RUN" = true ]; then
    print_warning "Would update discovery calls - manual review recommended"
else
    print_warning "Discovery API has changed - automatic migration is limited"
    print_warning "Please review the following files manually:"
    grep -rl "discover_devices()" . --include="*.py" 2>/dev/null | while read -r file; do
        echo "  - $file"
    done

    print_step "Note: Old API returns token IDs, new API returns device IDs"
    print_step "Consider using the compatibility layer for gradual migration"
fi

# Generate migration report
print_header "Step 6/6: Generating Migration Report"

REPORT_FILE="${SCRIPT_DIR}/migration_report_$(date +%Y%m%d_%H%M%S).txt"

cat > "$REPORT_FILE" << EOF
DSMIL Platform Migration Report
Generated: $(date)
=====================================

Migration Summary:
-----------------
$(if [ "$DRY_RUN" = true ]; then echo "DRY RUN - No changes made"; else echo "Changes applied successfully"; fi)
Backup created: $(if [ "$CREATE_BACKUP" = true ] && [ "$DRY_RUN" = false ]; then echo "$BACKUP_DIR"; else echo "No"; fi)

Files Modified:
--------------
Python files with updated imports: $(find . -name "*.py" -type f | wc -l)
Shell scripts with updated driver refs: $(find . -name "*.sh" -type f | wc -l)

Breaking Changes Applied:
------------------------
✓ Database imports updated (dsmil_device_database → dsmil_device_database_extended)
✓ Variable names updated (ALL_DEVICES → ALL_DEVICES_EXTENDED)
✓ Function names updated (get_device → get_device_extended)
✓ Control centre references updated
✓ Driver references updated (dsmil-84dev → dsmil-104dev)

Manual Review Required:
----------------------
! Discovery function calls - API signature changed
! Activation function calls - return type changed
! Any custom scripts using legacy APIs

Next Steps:
----------
1. Review changed files (check git diff if using version control)
2. Test the updated code:
   python3 dsmil.py diagnostics
3. Build and load the new driver:
   python3 dsmil.py build
   sudo python3 dsmil.py load
4. Run control centre:
   sudo python3 dsmil.py control
5. Review DEPRECATION_PLAN.md for detailed API changes

Compatibility Layer:
-------------------
For gradual migration, use the compatibility layer:
  from dsmil_legacy_compat import *

This provides backwards compatibility but is deprecated and will be
removed in v3.0.0 (2026 Q3).

Support:
-------
- Documentation: 00-documentation/
- Migration guide: DEPRECATION_PLAN.md
- Integration guide: 02-ai-engine/README_INTEGRATION.md

EOF

print_step "Migration report saved to: $REPORT_FILE"

# Final summary
echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}                     Migration Complete!                          ${BLUE}║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

if [ "$DRY_RUN" = true ]; then
    print_warning "This was a DRY RUN - no changes were made"
    print_warning "Run without --dry-run to apply changes"
else
    print_step "Your codebase has been migrated to v2.0 APIs"
    print_step "Backup created at: $BACKUP_DIR"
fi

echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "  1. Review the migration report: $REPORT_FILE"
echo "  2. Test your code: python3 dsmil.py diagnostics"
echo "  3. Build new driver: python3 dsmil.py build"
echo "  4. Load new driver: sudo python3 dsmil.py load"
echo "  5. Launch control centre: sudo python3 dsmil.py control"
echo ""
echo -e "${YELLOW}Important:${NC}"
echo "  - Review files manually for any missed updates"
echo "  - Discovery API returns device IDs (0-103) not token IDs"
echo "  - Activation API returns boolean not ActivationResult object"
echo "  - See DEPRECATION_PLAN.md for complete breaking changes list"
echo ""

if [ "$CREATE_BACKUP" = true ] && [ "$DRY_RUN" = false ]; then
    echo -e "${GREEN}Backup location:${NC} $BACKUP_DIR"
    echo ""
fi

exit 0

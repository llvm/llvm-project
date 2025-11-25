#!/usr/bin/env bash
#
# Manual Database Sync Script
# Syncs RAM disk database to persistent SQLite backup
#
# Usage:
#   ./sync_database.sh           # Normal sync
#   ./sync_database.sh --force   # Force sync even if no changes
#   ./sync_database.sh --restore # Restore from backup to RAM disk
#

set -e

RAMDISK_DB="/dev/shm/lat5150_ai/conversation_history.db"
BACKUP_DB="$(dirname "$0")/data/conversation_history.db"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if RAM disk DB exists
if [ ! -f "$RAMDISK_DB" ]; then
    print_error "RAM disk database not found: $RAMDISK_DB"
    print_warning "Database may not be running or RAM disk not mounted"
    exit 1
fi

# Create backup directory if needed
mkdir -p "$(dirname "$BACKUP_DB")"

# Check for --restore flag
if [ "$1" = "--restore" ]; then
    echo "Restoring database from backup..."

    if [ ! -f "$BACKUP_DB" ]; then
        print_error "Backup not found: $BACKUP_DB"
        exit 1
    fi

    # Copy backup to RAM disk
    cp -f "$BACKUP_DB" "$RAMDISK_DB"
    print_success "Restored from backup: $BACKUP_DB"
    print_success "Active database: $RAMDISK_DB"

    exit 0
fi

# Normal sync: RAM disk → Backup
echo "Syncing database to backup..."

# Get file sizes
RAMDISK_SIZE=$(stat -f "%z" "$RAMDISK_DB" 2>/dev/null || stat -c "%s" "$RAMDISK_DB")
RAMDISK_SIZE_MB=$(echo "scale=2; $RAMDISK_SIZE / 1024 / 1024" | bc)

# Check if force sync
if [ "$1" = "--force" ]; then
    FORCE=true
else
    FORCE=false
fi

# Sync to backup
if [ "$FORCE" = true ] || [ ! -f "$BACKUP_DB" ]; then
    cp -f "$RAMDISK_DB" "$BACKUP_DB"
    print_success "Database backed up: $BACKUP_DB"
else
    # Only copy if changed
    if ! cmp -s "$RAMDISK_DB" "$BACKUP_DB"; then
        cp -f "$RAMDISK_DB" "$BACKUP_DB"
        print_success "Database backed up: $BACKUP_DB (${RAMDISK_SIZE_MB} MB)"
    else
        print_warning "No changes detected, backup skipped"
        echo "  Use --force to sync anyway"
    fi
fi

# Show backup info
if [ -f "$BACKUP_DB" ]; then
    BACKUP_SIZE=$(stat -f "%z" "$BACKUP_DB" 2>/dev/null || stat -c "%s" "$BACKUP_DB")
    BACKUP_SIZE_MB=$(echo "scale=2; $BACKUP_SIZE / 1024 / 1024" | bc)
    BACKUP_TIME=$(stat -f "%Sm" "$BACKUP_DB" 2>/dev/null || stat -c "%y" "$BACKUP_DB" | cut -d'.' -f1)

    echo ""
    echo "Backup Details:"
    echo "  Location: $BACKUP_DB"
    echo "  Size: ${BACKUP_SIZE_MB} MB"
    echo "  Last modified: $BACKUP_TIME"
fi

exit 0

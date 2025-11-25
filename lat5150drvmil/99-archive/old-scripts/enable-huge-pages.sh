#!/bin/bash
# Enable Huge Pages for NPU Memory Manager
# Allocates 8GB of huge pages (4096 pages × 2MB each)

echo "🔧 Huge Pages Configuration for NPU"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check current status
echo "Current huge pages status:"
grep -i hugepages /proc/meminfo | grep -E "(HugePages_Total|Hugepagesize)"
echo ""

# Calculate pages needed for 8GB
# 8GB = 8192 MB
# Huge page size = 2MB
# Pages needed = 8192 / 2 = 4096
PAGES_NEEDED=4096

echo "NPU Memory Manager needs: 8GB"
echo "Huge page size: 2MB"
echo "Pages to allocate: $PAGES_NEEDED"
echo ""

# Check if we have enough free memory
TOTAL_MEM=$(grep MemTotal /proc/meminfo | awk '{print $2}')
TOTAL_MEM_GB=$((TOTAL_MEM / 1024 / 1024))

echo "System total memory: ${TOTAL_MEM_GB} GB"

if [ $TOTAL_MEM_GB -lt 16 ]; then
    echo "⚠️  Warning: System has < 16GB RAM"
    echo "   8GB huge pages may cause issues"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Allocating $PAGES_NEEDED huge pages..."

# Set huge pages
sudo sysctl -w vm.nr_hugepages=$PAGES_NEEDED

# Wait a moment for allocation
sleep 1

# Verify allocation
ALLOCATED=$(cat /proc/sys/vm/nr_hugepages)

if [ "$ALLOCATED" -eq "$PAGES_NEEDED" ]; then
    echo "✅ Successfully allocated $ALLOCATED huge pages"
    echo ""

    # Show new status
    echo "New huge pages status:"
    grep -i hugepages /proc/meminfo | grep -E "(HugePages_Total|HugePages_Free)"
    echo ""

    echo "✅ NPU Memory Manager should now work!"
    echo ""
    echo "To make permanent across reboots, add to /etc/sysctl.conf:"
    echo "  vm.nr_hugepages=$PAGES_NEEDED"
    echo ""
    echo "To test: cd /home/john/livecd-gen/npu_modules && ./bin/npu_memory_manager"
else
    echo "⚠️  Only allocated $ALLOCATED pages (requested $PAGES_NEEDED)"
    echo ""
    echo "Possible causes:"
    echo "  • Not enough contiguous memory available"
    echo "  • Memory fragmentation"
    echo "  • System under memory pressure"
    echo ""
    echo "Try:"
    echo "  1. Close applications to free memory"
    echo "  2. Run: sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'"
    echo "  3. Try again with fewer pages"
fi

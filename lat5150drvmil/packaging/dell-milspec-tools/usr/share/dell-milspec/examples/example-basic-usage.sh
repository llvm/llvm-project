#!/bin/bash
#
# Example: Basic Dell MIL-SPEC Tools Usage
# Demonstrates common operations with dell-milspec-tools package
#

set -e

echo "======================================================================"
echo " Dell MIL-SPEC Tools - Basic Usage Example"
echo "======================================================================"
echo ""

# Step 1: Check device status
echo "Step 1: Checking DSMIL device status..."
echo "Command: dsmil-status"
echo ""
dsmil-status
echo ""
read -p "Press Enter to continue..."
echo ""

# Step 2: Check TPM2 acceleration status
echo "Step 2: Checking TPM2 acceleration status..."
echo "Command: tpm2-accel-status"
echo ""
tpm2-accel-status
echo ""
read -p "Press Enter to continue..."
echo ""

# Step 3: Run basic device tests
echo "Step 3: Running basic device functionality tests..."
echo "Command: dsmil-test --basic-only"
echo ""
dsmil-test --basic-only
echo ""
read -p "Press Enter to continue..."
echo ""

# Step 4: Check configuration
echo "Step 4: Viewing configuration..."
echo ""
echo "DSMIL Configuration:"
cat /etc/dell-milspec/dsmil.conf | grep -v "^#" | grep -v "^$"
echo ""
read -p "Press Enter to continue..."
echo ""

# Step 5: Explain monitoring
echo "Step 5: Starting monitoring dashboard..."
echo ""
echo "The monitoring dashboard provides real-time system information:"
echo "  - System resource usage (CPU, memory, temperature)"
echo "  - DSMIL token state monitoring"
echo "  - Alert system for safety thresholds"
echo "  - Kernel message tracking"
echo ""
echo "Command: milspec-monitor"
echo ""
echo "Press Ctrl+C to stop monitoring when done."
read -p "Press Enter to start monitoring (or Ctrl+C to skip)..."
milspec-monitor --mode dashboard --duration 30

echo ""
echo "======================================================================"
echo " Example Complete"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  - Try dry-run token testing: dsmil-test --dry-run"
echo "  - Use control utility: milspec-control"
echo "  - View documentation in /usr/share/doc/dell-milspec-tools/"
echo ""

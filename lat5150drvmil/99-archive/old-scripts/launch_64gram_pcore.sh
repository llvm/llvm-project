#!/bin/bash
# Launch 64gram AVX-512 build on P-cores ONLY
# P-cores: CPUs 0-9 (have AVX-512)
# E-cores: CPUs 10-19 (NO AVX-512 - will crash)

TELEGRAM_BIN="/home/john/tdesktop/out/Release/bin/Telegram"

if [ ! -f "$TELEGRAM_BIN" ]; then
    echo "Error: 64gram binary not found at $TELEGRAM_BIN"
    echo "Build still in progress or failed."
    exit 1
fi

echo "========================================="
echo "Launching 64gram with AVX-512"
echo "Pinned to P-cores (CPUs 0-9)"
echo "========================================="
echo ""
echo "CPU Affinity: P-cores only (AVX-512 capable)"
echo "E-cores blocked (would crash on AVX-512)"
echo ""

# Use taskset to pin to P-cores (CPUs 0-9)
# Bitmask: 0x3FF = 0000001111111111 (first 10 CPUs)
taskset 0x3FF "$TELEGRAM_BIN" "$@"

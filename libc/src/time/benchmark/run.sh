#!/bin/bash
# Quick benchmark runner

set -e

cd "$(dirname "$0")"

echo "Building standalone benchmark..."
make -f Makefile.standalone clean > /dev/null 2>&1
make -f Makefile.standalone

echo ""
echo "Running benchmark..."
echo ""
./benchmark_time_conversion

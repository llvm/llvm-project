#!/bin/bash
# Simple build script for the time conversion benchmark

set -e

echo "Building time conversion benchmark..."

# Create build directory
BUILD_DIR="build_benchmark"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --target benchmark_time_conversion -j$(nproc)

echo ""
echo "Build complete! Run the benchmark with:"
echo "  ./$BUILD_DIR/benchmark_time_conversion"
echo ""
echo "Or run directly:"
cd ..
./"$BUILD_DIR"/benchmark_time_conversion

#!/usr/bin/env zsh
set -euo pipefail

ITERATIONS=2
TOOLS=("mlir-opt" "clang" "opt")
BUILD_DIR=build
TIMEFMT='user %U system %S cpu %P total %*E mem %M'

for cores in 32 16 8 4 2; do
for build_type in Release Debug; do
for unity_mode in off on; do
    if [ "$unity_mode" = "off" ]; then
        unity_flag=""
        label_suffix="$build_type - no unity"
    else
        unity_flag="-DCMAKE_UNITY_BUILD=ON"
        label_suffix="$build_type - with unity"
    fi

    for i in $(seq 1 $ITERATIONS); do

        rm -rf "$BUILD_DIR"

        cmake -S llvm -B "$BUILD_DIR" \
            -G Ninja \
            -DCMAKE_BUILD_TYPE="$build_type" \
            -DLLVM_ENABLE_PROJECTS="mlir;clang" \
            $unity_flag > /dev/null

        for tool in "${TOOLS[@]}"; do

            echo "========== cores $cores - $tool - $label_suffix - $i / $ITERATIONS =========="
            time_output=$( time ninja -j $cores -C "$BUILD_DIR" $tool 2>&1 > /dev/null )
            # time_output=$( time sleep 1 2>&1 > /dev/null )

            # Clean build artifacts but keep the configured build dir
            ninja -C "$BUILD_DIR" clean > /dev/null
        done

        rm -rf "$BUILD_DIR"
    done

done
done
done

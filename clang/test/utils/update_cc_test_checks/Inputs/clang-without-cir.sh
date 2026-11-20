#!/bin/sh
# Stands in for a clang built without CIR (CLANG_ENABLE_CIR=OFF).
echo "error: clang IR support not available, rebuild clang with -DCLANG_ENABLE_CIR=ON" >&2
exit 1

#!/bin/bash
# M3: memory model through the inter pipeline to the B60.
#   slm_kernel(out, in):    out[i] = in[tid] + in[rev] via SLM + barrier
#   atomic_kernel(out, ctr): out[i] = atomic old values, a permutation of 0..n
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=inter/out/m3

run_kernel() {
  local name=$1 ir=$2; shift 2
  local mode=$1; shift
  mkdir -p $OUT/$name
  inter/build/tools/inter-translate/inter-translate inter/test/Integration/$ir --import-llvm -o $OUT/$name/k.mlir
  inter/build/tools/inter-opt/inter-opt $OUT/$name/k.mlir \
    --inter-normalize-cf --lift-cf-to-scf --inter-convert-calls --inter-convert-memory --inter-select-to-machine --inter-insert-sync \
    -o $OUT/$name/k.xemachine.mlir
  inter/build/tools/inter-translate/inter-translate $OUT/$name/k.xemachine.mlir \
    --xemachine-to-zebin -o $OUT/$name/final.bin
  inter/out/launcher $OUT/$name/final.bin $name 128 "$@" | python3 inter/verify.py "$mode"
}

# out[gid] = in[gid] + in[gid & ~31 | (31 - lid)] = i + (i&~31)+31-(i&31)
run_kernel slm_kernel slm.ll 'i + (i & ~31) + 31 - (i & 31)' out in:1
run_kernel atomic_kernel atomic.ll --sort out in:0

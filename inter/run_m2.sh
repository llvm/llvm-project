#!/bin/bash
# M2: control flow through the inter pipeline to the B60.
#   branch_kernel(out, a, b, t):  out[i] = a[i] + (a[i] > t ? b[i] : 1)
#   uniform_kernel(out, a, b, t): out[i] = t > 3 ? a[i]+100 : b[i]
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=inter/out/m2

run_kernel() {
  local name=$1 cl=$2 expect=$3
  shift 3
  mkdir -p $OUT/$name/final
  clang-20 -target spir64-unknown-unknown -x cl -emit-llvm -S -O1 \
    -o $OUT/$name/k.ll inter/test/Integration/Inputs/$cl
  inter/build/tools/inter-translate/inter-translate $OUT/$name/k.ll --import-llvm -o $OUT/$name/k.mlir
  inter/build/tools/inter-opt/inter-opt $OUT/$name/k.mlir \
    --inter-normalize-cf --lift-cf-to-scf --inter-convert-calls --inter-convert-memory --inter-select-to-machine --inter-insert-sync \
    -o $OUT/$name/k.xemachine.mlir
  inter/build/tools/inter-translate/inter-translate $OUT/$name/k.xemachine.mlir \
    --xemachine-to-ged -o $OUT/$name/k.text.bin
  local ref=$OUT/$name/ref/${cl%.cl}_bmg.bin
  ocloc compile -file inter/test/Integration/Inputs/$cl -device bmg-g21 -out_dir $OUT/$name/ref -q
  python3 inter/make_zebin.py extract $ref $OUT/$name/final
  python3 inter/make_zebin.py write --kernel $name \
    --text $OUT/$name/k.text.bin \
    --zeinfo $OUT/$name/final/zeinfo.yaml \
    --notes $OUT/$name/final/note.compat.bin -o $OUT/$name/final.bin
  inter/out/launcher $OUT/$name/final.bin $name 128 "$@" | python3 inter/verify.py "$expect"
}

run_kernel branch_kernel branch.cl 'i + (1000*i if i > 5 else 1)' out in:1 in:1000 u32:5
run_kernel uniform_kernel uniform.cl 'i + 100 + 0*i' out in:1 in:1000 u32:7

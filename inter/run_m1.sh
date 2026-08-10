#!/bin/bash
# M1: vadd from clang LLVM IR through the inter pipeline to the B60.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=inter/out/m1
mkdir -p $OUT/final

clang-20 -target spir64-unknown-unknown -x cl -emit-llvm -S -o $OUT/vadd.ll inter/vadd.cl
inter/build/tools/inter-translate/inter-translate $OUT/vadd.ll --import-llvm -o $OUT/vadd.mlir
inter/build/tools/inter-opt/inter-opt $OUT/vadd.mlir --inter-normalize-cf --lift-cf-to-scf --inter-convert-calls --inter-convert-memory --inter-select-to-machine --inter-insert-sync -o $OUT/vadd.xemachine.mlir
inter/build/tools/inter-translate/inter-translate $OUT/vadd.xemachine.mlir --xemachine-to-ged -o $OUT/vadd.text.bin

# Reuse IGC's ABI metadata until the resource-info and zeinfo emitters land.
ocloc compile -file inter/vadd.cl -device bmg-g21 -out_dir $OUT/ref -q
python3 inter/make_zebin.py extract $OUT/ref/vadd_bmg.bin $OUT/final
python3 inter/make_zebin.py write --kernel vadd \
  --text $OUT/vadd.text.bin \
  --zeinfo $OUT/final/zeinfo.yaml \
  --notes $OUT/final/note.compat.bin -o $OUT/vadd_final.bin
inter/out/launcher $OUT/vadd_final.bin vadd 128 in:1 in:1000 out | python3 inter/verify.py 'i*1+i*1000'

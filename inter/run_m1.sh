#!/bin/bash
# M1: vadd from LLVM IR through the inter pipeline to the B60.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=inter/out/m1
mkdir -p $OUT

inter/build/tools/inter-translate/inter-translate inter/test/Integration/vadd.ll --import-llvm -o $OUT/vadd.mlir
inter/build/tools/inter-opt/inter-opt $OUT/vadd.mlir --inter-normalize-cf --lift-cf-to-scf --inter-convert-calls --inter-convert-memory --inter-select-to-machine --inter-insert-sync -o $OUT/vadd.xemachine.mlir
inter/build/tools/inter-translate/inter-translate $OUT/vadd.xemachine.mlir --xemachine-to-zebin -o $OUT/vadd.bin
inter/out/launcher $OUT/vadd.bin vadd 128 in:1 in:1000 out | python3 inter/verify.py 'i*1+i*1000'

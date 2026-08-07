#!/bin/bash
# M1: vadd from clang LLVM IR through the inter pipeline to the B60.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=inter/out/m1
B=build-m0
mkdir -p $OUT/dump

clang-20 -target spir64-unknown-unknown -x cl -emit-llvm -S -o $OUT/vadd.ll inter/vadd.cl
inter/build/tools/inter-translate/inter-translate $OUT/vadd.ll --import-llvm -o $OUT/vadd.mlir
inter/build/tools/inter-opt/inter-opt $OUT/vadd.mlir --inter-normalize-cf --lift-cf-to-scf --inter-select-to-machine -o $OUT/vadd.xemachine.mlir
inter/build/tools/inter-translate/inter-translate $OUT/vadd.xemachine.mlir --xemachine-to-iga -o $OUT/vadd.asm

# Assemble via ocloc (needs the dump-dir structure; reuse the reference
# container metadata).
if [ ! -d $OUT/refdump ]; then
  ocloc compile -file inter/vadd.cl -device bmg-g21 -out_dir $OUT/ref -q
  ocloc disasm -file $OUT/ref/vadd_bmg.bin -device bmg-g21 -dump $OUT/refdump
fi
cp $OUT/vadd.asm $OUT/dump/.text.vadd.asm
cp $OUT/refdump/.??* $OUT/refdump/sections.txt $OUT/dump/
(cd $OUT && ocloc asm -file vadd_inter.bin -dump dump -device bmg-g21)

# Our container + generic launcher; verify on hardware.
mkdir -p $OUT/final
python3 inter/make_zebin.py extract $OUT/dump/vadd_inter.bin $OUT/final
python3 inter/make_zebin.py write --kernel vadd \
  --text $OUT/final/vadd.text.bin \
  --zeinfo $OUT/final/zeinfo.yaml \
  --notes $OUT/final/note.compat.bin -o $OUT/vadd_final.bin
inter/out/launcher $OUT/vadd_final.bin vadd 128 in:1 in:1000 out | python3 inter/verify.py 'i*1+i*1000'

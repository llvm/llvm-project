#!/bin/bash
# M3: memory model through the inter pipeline to the B60.
#   slm_kernel(out, in):    out[i] = in[tid] + in[rev] via SLM + barrier
#   atomic_kernel(out, ctr): out[i] = atomic old values, a permutation of 0..n
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=inter/out/m3

run_kernel() {
  local name=$1 cl=$2; shift 2
  local mode=$1; shift
  mkdir -p $OUT/$name/dump
  clang-20 -target spir64-unknown-unknown -x cl -emit-llvm -S -O1 \
    -o $OUT/$name/k.ll inter/$cl
  inter/build/tools/inter-translate/inter-translate $OUT/$name/k.ll --import-llvm -o $OUT/$name/k.mlir
  inter/build/tools/inter-opt/inter-opt $OUT/$name/k.mlir \
    --inter-normalize-cf --lift-cf-to-scf --inter-select-to-machine \
    -o $OUT/$name/k.xemachine.mlir
  inter/build/tools/inter-translate/inter-translate $OUT/$name/k.xemachine.mlir \
    --xemachine-to-iga -o $OUT/$name/k.asm
  if [ ! -d $OUT/$name/refdump ]; then
    ocloc compile -file inter/$cl -device bmg-g21 -out_dir $OUT/$name/ref -q
    ocloc disasm -file $OUT/$name/ref/${cl%.cl}_bmg.bin -device bmg-g21 -dump $OUT/$name/refdump
  fi
  cp $OUT/$name/k.asm $OUT/$name/dump/.text.$name.asm
  cp $OUT/$name/refdump/.??* $OUT/$name/refdump/sections.txt $OUT/$name/dump/
  (cd $OUT/$name && ocloc asm -file final.bin -dump dump -device bmg-g21) > /dev/null
  mkdir -p $OUT/$name/final
  python3 inter/make_zebin.py extract $OUT/$name/dump/final.bin $OUT/$name/final
  python3 inter/make_zebin.py write --kernel $name \
    --text $OUT/$name/final/$name.text.bin \
    --zeinfo $OUT/$name/final/zeinfo.yaml \
    --notes $OUT/$name/final/note.compat.bin -o $OUT/$name/final.bin
  inter/out/launcher $OUT/$name/final.bin $name 128 "$@" | python3 inter/verify.py "$mode"
}

# out[gid] = in[gid] + in[gid & ~31 | (31 - lid)] = i + (i&~31)+31-(i&31)
run_kernel slm_kernel slm.cl 'i + (i & ~31) + 31 - (i & 31)' out in:1
run_kernel atomic_kernel atomic.cl --sort out in:0

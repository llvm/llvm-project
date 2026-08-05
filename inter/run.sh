#!/bin/bash
# Container proof, end to end. Run from anywhere; regenerates inter/out/.
# Stages: (1) ocloc reference zebin, (2) our zebin writer, (3) hand-edited asm.
set -euo pipefail
cd "$(dirname "$0")/.."

B=build-m0
OUT=inter/out
K=scale

rm -rf $OUT
mkdir -p $OUT/ref $OUT/extracted $OUT/mod $OUT/mod2x

# liboffload with the level_zero plugin (excluded from "all" on Linux; must be
# requested explicitly). BUILD_LIBOMPTARGET=OFF needs the local
# offload/CMakeLists.txt option patch; libomptarget drags in openmp, which
# does not configure in this tree.
if [ ! -f $B/lib/x86_64-unknown-linux-gnu/libLLVMOffload.so ]; then
  cmake -S llvm -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_RUNTIMES=offload -DLLVM_INCLUDE_TESTS=OFF \
    -DRUNTIMES_CMAKE_ARGS="-DOFFLOAD_INCLUDE_TESTS=OFF;-DBUILD_LIBOMPTARGET=OFF;-DLIBOMPTARGET_PLUGINS_TO_BUILD=level_zero"
fi
ninja -C $B offload

c++ -std=c++17 -O1 -o $OUT/launcher inter/launcher.cpp \
  -I $B/runtimes/runtimes-bins/offload/liboffload/API \
  -L $B/lib/x86_64-unknown-linux-gnu -lLLVMOffload \
  -Wl,-rpath,$PWD/$B/lib/x86_64-unknown-linux-gnu

# Reference kernel via ocloc (IGC) for bmg-g21.
ocloc compile -file inter/$K.cl -device bmg-g21 -out_dir $OUT/ref

# Stage 1: host path proof with the reference zebin.
$OUT/launcher $OUT/ref/${K}_bmg.bin $K 128 7 2

# Stage 2: same kernel bytes in our own zebin container.
python3 inter/make_zebin.py extract $OUT/ref/${K}_bmg.bin $OUT/extracted
python3 inter/make_zebin.py write --kernel $K \
  --text $OUT/extracted/$K.text.bin \
  --zeinfo $OUT/extracted/zeinfo.yaml \
  --notes $OUT/extracted/note.compat.bin -o $OUT/ours.bin
$OUT/launcher $OUT/ours.bin $K 128 7 2

# Stage 3: hand-edited asm (shl 1 -> 2) reassembled by ocloc, our container.
# Note: this ocloc build writes the assembled binary into the dump dir and
# only accepts -file for the output name; -out is broken.
ocloc disasm -file $OUT/ref/${K}_bmg.bin -device bmg-g21 -dump $OUT/mod
sed -i 's/r8.0<1;1,0>:d     1:w/r8.0<1;1,0>:d     2:w/' $OUT/mod/.text.$K.asm
(cd $OUT && ocloc asm -file mod.bin -dump mod -device bmg-g21)
python3 inter/make_zebin.py extract $OUT/mod/mod.bin $OUT/mod2x
python3 inter/make_zebin.py write --kernel $K \
  --text $OUT/mod2x/$K.text.bin \
  --zeinfo $OUT/mod2x/zeinfo.yaml \
  --notes $OUT/mod2x/note.compat.bin -o $OUT/mod_ours.bin
$OUT/launcher $OUT/mod_ours.bin $K 128 7 4

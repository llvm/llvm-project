// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -g -S -fenable-ripple -O2 -emit-llvm %s -o - 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"
// RUN: %clang -x c++ -g -S -fenable-ripple -O2 -emit-llvm %s -o - 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"

#include <ripple.h>

const size_t ThisIsAGlobalThatCanBeUsedForRippleIndexing = 42;

size_t validIndexingFun(size_t BlockIndex, size_t BlockSize) {
  return (BlockIndex + ThisIsAGlobalThatCanBeUsedForRippleIndexing) % BlockSize;
}

// CHECK: valid_shuffle_index_using_const_global
void valid_shuffle_index_using_const_global(size_t ArraySize, float *Input,
                                            float *Output) {
  ripple_block_t BS = ripple_set_block_shape(0, 1);
  unsigned BlockIdx = ripple_id(BS, 0);
  unsigned BlockSize = ripple_get_block_size(BS, 0);
  for (size_t Idx = 0; Idx < ArraySize; Idx += BlockSize) {
    float Tmp = Input[Idx + BlockIdx];
    float Shuffled = ripple_shuffle(Tmp, validIndexingFun);
    Output[Idx + BlockIdx] = Shuffled;
  }
}

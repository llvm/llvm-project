// This checks without "-g"
// REQUIRES: aarch64-registered-target || x86-registered-target
// RUN: %clang -S -O1 -emit-llvm -fenable-ripple %s 2> %t; FileCheck %s --input-file %t

#include <ripple.h>

const size_t ThisIsAGlobalThatCanBeUsedForRippleIndexing = 42;

size_t indexingFun(size_t BlockIndex, size_t BlockSize) {
  // The issue is OOB since we are returning values >= 42 for a vector of size 3!
  return BlockIndex + ThisIsAGlobalThatCanBeUsedForRippleIndexing;
}

void oob_shuffle_fail_without_debuginfo(size_t ArraySize, float *Input,
                                        float *Output) {
  ripple_block_t BS = ripple_set_block_shape(0, 3);
  unsigned BlockIdx = ripple_id(BS, 0);
  unsigned BlockSize = ripple_get_block_size(BS, 0);
  for (size_t Idx = 0; Idx < ArraySize; Idx += BlockSize) {
    float Tmp = Input[Idx + BlockIdx];
    float Shuffled = ripple_shuffle(Tmp, indexingFun);
    Output[Idx + BlockIdx] = Shuffled;
  }
}

// CHECK: Evaluation of the index mapping function of ripple_shuffle returned an out of bound value; the call was to "indexingFun" with arguments (i{{[0-9]+}} 0, i{{[0-9]+}} 3). The returned value (42) is greater or equal to the size of the tensor (3)

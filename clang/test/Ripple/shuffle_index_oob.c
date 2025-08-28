// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -g -S -fenable-ripple -O2 -emit-llvm %s 2> %t; FileCheck %s --input-file %t
// RUN: %clang -x c++ -g -S -fenable-ripple -O2 -emit-llvm %s 2> %t; FileCheck %s --input-file %t

#include <ripple.h>

size_t indexingFun(size_t BlockIndex, size_t BlockSize) {
  return BlockIndex - BlockSize;
}

void shuffle_index_oob_test(size_t ArraySize, float *Input, float *Output) {
  ripple_block_t BS = ripple_set_block_shape(0, 2);
  unsigned BlockIdx = ripple_id(BS, 0);
  unsigned BlockSize = ripple_get_block_size(BS, 0);
  for (size_t Idx = 0; Idx < ArraySize; Idx += BlockSize) {
    float Tmp = Input[Idx + BlockIdx];
    // CHECK: [[@LINE+1]]:{{[0-9]+}}: error: Evaluation of the index mapping function of ripple_shuffle returned an out of bound value
    float Shuffled = ripple_shuffle(Tmp, indexingFun);
    Output[Idx + BlockIdx] = Shuffled;
  }
}

// CHECK: error: Ripple failed to vectorize this function

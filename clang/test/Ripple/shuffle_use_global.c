// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -g -S -fenable-ripple -O2 -emit-llvm %s 2> %t; FileCheck %s --input-file %t
// RUN: %clang -x c++ -g -S -fenable-ripple -O2 -emit-llvm %s 2> %t; FileCheck %s --input-file %t

#include <ripple.h>

size_t ThisIsAGlobalAndCannotBeUsedForRippleIndexing = 42;

size_t invalidIndexingFun(size_t BlockIndex, size_t BlockSize) {
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: error: The ripple shuffle instruction index-mapping function (or lambda) cannot be evaluated at compile time because it is accessing the value of a non-constant global variable
  return BlockIndex - BlockSize + ThisIsAGlobalAndCannotBeUsedForRippleIndexing;
}

void invalid_indexing_shuffle(size_t ArraySize, float *Input, float *Output) {
  ripple_block_t BS = ripple_set_block_shape(0, 2);
  unsigned BlockIdx = ripple_id(BS, 0);
  unsigned BlockSize = ripple_get_block_size(BS, 0);
  for (size_t Idx = 0; Idx < ArraySize; Idx += BlockSize) {
    float Tmp = Input[Idx + BlockIdx];
    float Shuffled = ripple_shuffle(Tmp, invalidIndexingFun);
    Output[Idx + BlockIdx] = Shuffled;
  }
}

// CHECK: error: Ripple failed to vectorize this function

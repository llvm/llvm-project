// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -g -S -fenable-ripple -O2 -emit-llvm %s 2> %t; FileCheck %s --input-file %t
// RUN: %clang -x c++ -g -S -fenable-ripple -O2 -emit-llvm %s 2> %t; FileCheck %s --input-file %t

#include <ripple.h>

#ifdef __cplusplus
extern "C"
#endif
    size_t
    indexingFun(size_t BlockIndex, size_t BlockSize) {
  return BlockSize * 2;
}

void shuffle_index_oob_test(size_t ArraySize, float *Input, float *Input2,
                            float *Output) {
  ripple_block_t BS = ripple_set_block_shape(0, 2);
  unsigned BlockIdx = ripple_id(BS, 0);
  unsigned BlockSize = ripple_get_block_size(BS, 0);
  for (size_t Idx = 0; Idx < ArraySize; Idx += BlockSize) {
    float Tmp = Input[Idx + BlockIdx];
    float Tmp2 = Input2[Idx + BlockIdx];
    // CHECK: shuffle_pair_index_oob.c:25:{{.*}}Evaluation of the index mapping function of ripple_shuffle_pair returned an out of bound value; the call was to "indexingFun" with arguments (i{{[0-9]+}} 0, i{{[0-9]+}} 2). The returned value (4) is greater or equal to the size of two (pair) tensors (4)

    float Shuffled = ripple_shuffle_pair(Tmp, Tmp2, indexingFun);
    Output[Idx + BlockIdx] = Shuffled;
  }
}

// CHECK: shuffle_pair_index_oob.c:16:{{.*}}Ripple failed to vectorize this function

// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -g -S -fenable-ripple -O2 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang -x c++ -g -S -fenable-ripple -O2 -emit-llvm %s -o - | FileCheck %s

#include <ripple.h>

#ifdef __cplusplus
extern "C"
#endif
    size_t
    indexingFun(size_t BlockIndex, size_t BlockSize) {
  return BlockSize * 2 - 1;
}

// CHECK: shuffle_index_max_test
// CHECK: %[[LoadIn2:[0-9A-Za-z_.]+]] = load <32 x float>, ptr %Input2
// CHECK: shufflevector <32 x float> %[[LoadIn2]], <32 x float> poison, <32 x i{{[0-9]+}}> <i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31, i{{[0-9]+}} 31>
void shuffle_index_max_test(size_t ArraySize, float *Input, float *Input2,
                            float *Output) {
  ripple_block_t BS = ripple_set_block_shape(0, 32);
  size_t Idx = ripple_id(BS, 0);
  size_t BlockSize = ripple_get_block_size(BS, 0);
  float Tmp = Input[Idx];
  float Tmp2 = Input2[Idx];
  float Shuffled = ripple_shuffle_pair(Tmp, Tmp2, indexingFun);
  Output[Idx] = Shuffled;
}

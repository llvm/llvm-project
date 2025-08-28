// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -g -S -fenable-ripple -O2 -emit-llvm %s 2> %t; echo %t; FileCheck %s --input-file %t
// RUN: %clang -x c++ -g -S -fenable-ripple -O2 -emit-llvm %s 2> %t; cat %t; FileCheck %s --input-file %t

// Testing that calling an external function creates a warning
#include <ripple.h>

extern size_t externalIndexingFun(size_t, size_t);

void invalid_external_shuffle_func_test(size_t ArraySize, float *Input,
                                        float *Output) {
  ripple_block_t BS = ripple_set_block_shape(0, 4);
  unsigned BlockIdx = ripple_id(BS, 0);
  unsigned BlockSize = ripple_get_block_size(BS, 0);
  for (size_t Idx = 0; Idx < ArraySize; Idx += BlockSize) {
    float Tmp = Input[Idx + BlockIdx];

    float Shuffled = ripple_shuffle(Tmp, externalIndexingFun);
    Output[Idx + BlockIdx] = Shuffled;
  }
}

// CHECK: {{.*}}shuffle_external.c:18:22: error: the index mapping function (or lambda) operand of ripple shuffle requires its definition to be accessible in the same module as the function being processed

// CHECK: error: Ripple failed to vectorize this function

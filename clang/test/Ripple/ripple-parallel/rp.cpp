// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -Xclang -disable-llvm-passes -S -emit-llvm -fenable-ripple %s -o - | FileCheck %s

#include <ripple.h>

// Here we're checking that the transformation happens for C++ files as well

template <typename Ttype>
void elementwise_add_ripple(int length, Ttype *outptr, const Ttype *aptr,
                            const Ttype *bptr) {
  ripple_block_t BS = ripple_set_block_shape(0, 1024 / sizeof(Ttype));
  // CHECK: %ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
  ripple_parallel(BS, 0);
  for (size_t i = 0; i < length; i++) {
      // partial_sum gets expanded automatically to a 1-d block of floats
      outptr[i] = aptr[i] + bptr[i];
  }
}

void foo(int n, float * C, float * A, float * B) {
    elementwise_add_ripple(n, C, A, B);
}

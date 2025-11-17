// REQUIRES: hexagon-registered-target
// RUN: %clang -S --target=hexagon -mhvx -mv81 -mhvx-length=128B -O2 -fenable-ripple -fdisable-ripple-lib -mllvm -ripple-pad-to-target-simd -emit-llvm %s -o - 2>&1 | FileCheck %s

#include <ripple.h>

void check_that_phis_are_padded(size_t N, int8_t a[N], int8_t b[N], int8_t apb[N]) {
  ripple_block_t BS = ripple_set_block_shape(0, 127);
  size_t v0 = ripple_id(BS, 0);

  size_t my_val = 0;
  for (int i = 0; i < N; i+= 127) {
    my_val += v0*v0;
    apb[i+v0] = my_val * (a[i+v0] + b[i+v0]);
  }
}

// CHECK: check_that_phis_are_padded
// CHECK: phi <128 x i32>
// CHECK: tail call <128 x i8> @llvm.masked.load.v128i8.p0
// CHECK: tail call <128 x i8> @llvm.masked.load.v128i8.p0
// CHECK: tail call void @llvm.masked.store.v128i8.p0

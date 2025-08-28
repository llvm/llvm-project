// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -Wall -Wextra -Wpedantic -fenable-ripple -S -O2 -emit-llvm %s -o - | FileCheck --implicit-check-not="warning:" %s

#include <ripple.h>

__attribute__((noinline)) int toBeSpecialized(int n) {
  return n * 32;
}

// CHECK-LABEL: @checkSize4
// CHECK: call fastcc <4 x i{{[0-9]+}}> @ripple.specialization.final.{{[0-9]+}}.toBeSpecialized(<4 x i{{[0-9]+}}

// CHECK-LABEL: @checkSize8
// CHECK: call fastcc <8 x i{{[0-9]+}}> @ripple.specialization.final.{{[0-9]+}}.toBeSpecialized(<8 x i{{[0-9]+}}

// CHECK-LABEL: define{{.*}}internal{{.*}}fastcc{{.*}}<8 x i{{[0-9]+}}> @ripple.specialization.final.{{[0-9]+}}.toBeSpecialized(<8 x i{{[0-9]+}}>{{.*}}%0)
// CHECK: %[[MulBy32:[A-Za-z0-9_.]+]] = shl <8 x i{{[0-9]+}}> %0, splat (i{{[0-9]+}} 5)
// CHECK-NEXT: ret <8 x i{{[0-9]+}}> %[[MulBy32]]

// CHECK-LABEL: define{{.*}}internal{{.*}}fastcc{{.*}}<4 x i{{[0-9]+}}> @ripple.specialization.final.{{[0-9]+}}.toBeSpecialized(<4 x i{{[0-9]+}}>{{.*}}%0)
// CHECK: %[[MulBy32:[A-Za-z0-9_.]+]] = shl <4 x i{{[0-9]+}}> %0, splat (i{{[0-9]+}} 5)
// CHECK-NEXT: ret <4 x i{{[0-9]+}}> %[[MulBy32]]

void checkSize4(int *in, int *out) {
  ripple_block_t BS = ripple_set_block_shape(0, 4);
  size_t idx = ripple_id(BS, 0);
  out[idx] = toBeSpecialized(in[idx]);
}

void checkSize8(int *in, int *out) {
  ripple_block_t BS = ripple_set_block_shape(0, 8);
  size_t idx = ripple_id(BS, 0);
  out[idx] = toBeSpecialized(in[idx]);
}

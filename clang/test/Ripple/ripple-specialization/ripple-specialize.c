// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -Wall -Wextra -Wpedantic -fenable-ripple -S -O2 -emit-llvm %s -o - | FileCheck --implicit-check-not="warning:" %s

#include <ripple.h>

__attribute__((noinline)) int toBeSpecialized(int n) {
  return n * 32;
}

// CHECK-LABEL: @test
// CHECK: call fastcc <128 x i{{[0-9]+}}> @ripple.specialization.final.{{[0-9]+}}.toBeSpecialized(<128 x i{{[0-9]+}}>

// CHECK-LABEL: define{{.*}}internal{{.*}}fastcc{{.*}}<128 x i{{[0-9]+}}> @ripple.specialization.final.{{[0-9]+}}.toBeSpecialized(<128 x i{{[0-9]+}}>{{.*}}%0)
// CHECK: %[[MulBy32:[A-Za-z0-9_.]+]] = shl <128 x i{{[0-9]+}}> %0, splat (i{{[0-9]+}} 5)
// CHECK-NEXT: ret <128 x i{{[0-9]+}}> %[[MulBy32]]

void test(int in[128], int *out) {
  ripple_block_t BS = ripple_set_block_shape(0, 128);
  size_t idx = ripple_id(BS, 0);
  out[idx] = toBeSpecialized(in[idx]);
}

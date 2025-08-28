// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -Wall -Wextra -Wpedantic -fenable-ripple -S -O2 -emit-llvm %s -o - | FileCheck --implicit-check-not="warning:" %s

#include <ripple.h>

__attribute__((noinline)) int toBeSpecialized(int n) {
  return ripple_reducemax(0x1, n * 32);
}

// CHECK: @test
// CHECK: call fastcc i{{[0-9]+}} @ripple.specialization.final.{{[0-9]+}}.toBeSpecialized(<128 x i{{[0-9]+}}>

// CHECK: define{{.*}}internal{{.*}}fastcc{{.*}}i{{[0-9]+}} @ripple.specialization.final.{{[0-9]+}}.toBeSpecialized(<128 x i{{[0-9]+}}>{{.*}}%0)

void test(int in[128], int *out) {
  ripple_block_t BS = ripple_set_block_shape(0, 128);
  size_t idx = ripple_id(BS, 0);
  *out = toBeSpecialized(in[idx]);
}


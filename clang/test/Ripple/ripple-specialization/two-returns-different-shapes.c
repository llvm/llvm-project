// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -g -Wall -Wextra -Wpedantic -fenable-ripple -S -O2 -emit-llvm %s -o - | FileCheck --implicit-check-not="warning:" %s

#include <ripple.h>

static __attribute__((noinline)) int toBeSpecialized(int n, int m) {
  if (n < 32)
    return n;
  else
    return m;
}

void test(int *in, int*out) {
  ripple_block_t BS = ripple_set_block_shape(0, 4, 8);
  size_t idx0 = ripple_id(BS, 0);
  size_t idx1 = ripple_id(BS, 1);
  out[idx0 + idx1] = toBeSpecialized(in[idx0], in[idx1]);
}

// CHECK: tail call fastcc <32 x i{{[0-9]+}}> @ripple.specialization.final.1.toBeSpecialized(<4 x i{{[0-9]+}}> %{{.*}}, <8 x i{{[0-9]+}}>
// CHECK: define{{.*}}fastcc{{.*}}<32 x i{{[0-9]+}}> @ripple.specialization.final.1.toBeSpecialized(<4 x i{{[0-9]+}}>{{.*}}, <8 x i{{[0-9]+}}>

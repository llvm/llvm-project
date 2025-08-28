// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -g -Wall -Wextra -Wpedantic -fenable-ripple -S -O2 -emit-llvm %s -o - | FileCheck --implicit-check-not="warning:" %s

#include <ripple.h>

__attribute__((noinline)) static void toBeSpecialized1(ripple_block_t, int);
__attribute__((noinline)) static void toBeSpecialized2(int, ripple_block_t);
__attribute__((noinline)) static void toBeSpecialized3(ripple_block_t, int);

extern int *somewhere;
extern int externalEnding;

void toBeSpecialized1(ripple_block_t BS, int n) {
  if (++externalEnding < 5)
    return;
  toBeSpecialized2(n, BS);
  somewhere[ripple_id(BS, 0)] += n;
}

void toBeSpecialized2(int n, ripple_block_t BS) {
  somewhere[ripple_id(BS,0)] *= n;
  toBeSpecialized3(BS, n);
  toBeSpecialized1(BS, n);
}

void toBeSpecialized3(ripple_block_t BS, int n) {
  toBeSpecialized1(BS, n + 32);
}

void test(int *in) {
  ripple_block_t BS = ripple_set_block_shape(0, 4);
  size_t idx = ripple_id(BS, 0);
  toBeSpecialized1(BS, in[idx]);
}

// CHECK: test
// CHECK: tail call fastcc void @ripple.specialization.final.{{[0-9]+}}.toBeSpecialized1(ptr poison, <4 x i32>
// CHECK: define{{.*}}fastcc void @ripple.specialization.final.{{[0-9]+}}.toBeSpecialized1(ptr %{{.*}}, <4 x i32>
// CHECK: define{{.*}}fastcc void @ripple.specialization.final.{{[0-9]+}}.toBeSpecialized2(<4 x i32>{{.*}}, ptr
// CHECK: define{{.*}}fastcc void @ripple.specialization.final.{{[0-9]+}}.toBeSpecialized3(ptr %{{.*}}, <4 x i32>

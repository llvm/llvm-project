// REQUIRES: aarch64-registered-target
// RUN: %clang -ffreestanding --target=aarch64-unknown-linux-gnu -S -O2 -emit-llvm %s -DCOMPILE_LIB=1 -o %t.ll
// RUN: %clang -ffreestanding -g --target=aarch64-unknown-linux-gnu -S -O2 -emit-llvm -fenable-ripple -fripple-lib=%t.ll -mllvm -ripple-disable-link %s 2>%t.err; FileCheck %s --input-file=%t.err

// Checks that we get an error for externf2 usage: declaration using ripple_block_t

#ifdef COMPILE_LIB

typedef float v4f32 __attribute__((__vector_size__(16)))
__attribute__((aligned(16)));
typedef float v16f32 __attribute__((__vector_size__(64)))
__attribute__((aligned(16)));

extern v4f32 ripple_externf1(void) { v4f32 v = {10.f, 12.f}; return v; }
extern v4f32 ripple_externf3(void) { v4f32 v = {10.f, 12.f}; return v; }

#else

#include <ripple.h>

extern float externf1(ripple_block_t BS);
extern float externf2(ripple_block_t BS);
extern float externf3(ripple_block_t BS, float);

void test_invalid_candidates(float *restrict f, float *restrict f2, float *restrict f3) {
  ripple_block_t BS = ripple_set_block_shape(0, 4, 4);
  size_t RID = ripple_id(BS, 0);
  size_t RID2 = ripple_id(BS, 1);

  f[RID] = externf1(BS);
  f[RID2] += externf2(BS);
}

void test_invalid_candidates_2(float *restrict f, float *restrict f2, float *restrict f3) {
  ripple_block_t BS = ripple_set_block_shape(0, 4, 4);
  size_t RID = ripple_id(BS, 0);
  size_t RID2 = ripple_id(BS, 1);

  f[RID] = externf1(BS);
  f[RID2] += externf3(BS, 42.f);
}

#endif // COMPILE_LIB

// CHECK:       external-void-absent.c:40:14: error: Passing a ripple block shape to a function call with no known definition is not allowed. Make sure that the function is available for ripple processing.
// CHECK-NEXT:  40 |   f[RID2] += externf3(BS, 42.f);
// CHECK-NEXT:     |              ^
// CHECK:       external-void-absent.c:31:14: error: Passing a ripple block shape to a function call with no known definition is not allowed. Make sure that the function is available for ripple processing.
// CHECK-NEXT:  31 |   f[RID2] += externf2(BS);
// CHECK-NEXT:     |              ^

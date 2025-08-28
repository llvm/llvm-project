// REQUIRES: target-x86_64 || target=hexagon{{.*}}
// RUN: %clang -g -c -O2 -emit-llvm %s -DCOMPILE_LIB=1 -o %t.bc
// RUN: %clang -g -O2 -fenable-ripple -fripple-lib=%t.bc -emit-llvm -S %s -o - | FileCheck %s

#ifdef COMPILE_LIB

#include <stdint.h>

typedef float f32t32 __attribute__((__vector_size__(128)))
__attribute__((aligned(128)));

extern inline f32t32 ripple_pure_ew_add(f32t32 A) {
  return (A + 2.f);
}

#else

#include <stddef.h>
#include <ripple.h>

#define VEC 0

extern float add(float);

// CHECK: call{{.*}}@ripple_pure_ew_add(
// CHECK: call float @add(float

// Checks that the external function is not called for the scalar "add"
void check(size_t size, float *input, float *output) {
  ripple_block_t BS = ripple_set_block_shape(VEC, 32);
  size_t BlockX = ripple_id(BS, 0);
  size_t BlockSizeX = ripple_get_block_size(BS, 0);
  output[0] = ripple_reduceadd(0x1, add(input[BlockX])) + add(42.f);;
}

#endif

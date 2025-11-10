// REQUIRES: target-x86_64 || target=hexagon{{.*}}
// RUN: %clang -g -c -O2 -emit-llvm %s -DCOMPILE_LIB=1 -o %t.rlib.bc
// RUN: %clang -g -O2 -fenable-ripple -fripple-lib=%t.rlib.bc -emit-llvm -S %s -o - -mllvm -ripple-disable-link | FileCheck %s

#ifdef COMPILE_LIB

#include <stdint.h>

typedef int32_t i32t32 __attribute__((__vector_size__(128)))
__attribute__((aligned(128)));

extern inline void ripple_f(int32_t *a, int32_t *b, i32t32 c) {
  *((i32t32 *)a) = c + *((i32t32 *)b);
}

extern inline void ripple_mask_f(int32_t *a, int32_t *b, i32t32 c, i32t32 d) {
  *((i32t32 *)a) = c + d + *((i32t32 *)b);
}

#else

#include <ripple.h>
#include <stddef.h>

#define VEC 0

extern void f(int32_t *, int32_t *, int32_t);

// CHECK: @test
// CHECK: call void @ripple_f
// CHECK: call void @ripple_mask_f
void test(size_t size, int32_t *input, int32_t *output) {
  ripple_block_t BS = ripple_set_block_shape(VEC, 32);
  size_t BlockX = ripple_id(BS, 0);
  size_t BlockSizeX = ripple_get_block_size(BS, 0);
  ripple_parallel(BS, 0);
  for (int32_t i = 0; i < size; i++)
    f(input, output, i);
}

#endif

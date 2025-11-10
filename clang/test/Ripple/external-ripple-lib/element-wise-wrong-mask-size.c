// REQUIRES: target-x86_64 || target=hexagon{{.*}}
// RUN: %clang -g -c -O2 -emit-llvm %s -DCOMPILE_LIB=1 -o %t.bc
// RUN: %clang -g -O2 -fenable-ripple -fripple-lib=%t.bc -emit-llvm -S -mllvm -ripple-disable-link %s 2> %t; FileCheck %s --input-file %t

#ifdef COMPILE_LIB

#include <stdint.h>

typedef float f32t32 __attribute__((__vector_size__(128)))
__attribute__((aligned(128)));

typedef signed char i1t16 __attribute__((__vector_size__(16)))
__attribute__((aligned(16)));

extern inline f32t32 ripple_mask_add(f32t32 A, f32t32 B,
                                     i1t16 Mask) {
  (void)Mask;
  return ((A + B) / 2.f);
}

#else

#include <stddef.h>
#include <ripple.h>

#define VEC 0

extern float add(float, float);

// CHECK: the mask operand of the external ripple function 'ripple_mask_add' must be compatible with its return and non-mask operand shapes, i.e., all operands and return tensor shape can be broadcasted to the mask shape: expected mask of size 32 but have <16 x i8>

void ew_smaller(size_t size, _Float16 *input, _Float16 *output) {
  ripple_block_t BS = ripple_set_block_shape(VEC, 32);
  size_t BlockX = ripple_id(BS, 0);
  size_t BlockSizeX = ripple_get_block_size(BS, 0);
  for (size_t i = 0; i < size; i += BlockSizeX)
    if (i + BlockX < size)
      output[i + BlockX] = add(input[i + BlockX], input[i + BlockX]);
}

#endif

// REQUIRES: target-x86_64 || target=hexagon{{.*}}
// RUN: %clang -g -c -O2 -emit-llvm %S/external_library.c -o %t.rlib.bc
// RUN: %clang -g -O2 -fenable-ripple -fripple-lib=%t.rlib.bc -emit-llvm -S %s 2> %t.err; FileCheck %s --input-file %t.err

#include <stddef.h>
#include <ripple.h>

#define VEC 0

float ripple_add_and_half_ptr(float a, float*ptr) {
  return a + *ptr;
}

// CHECK: A Ripple symbol with name "ripple_add_and_half_ptr" is already defined in the current module with type "float (float, ptr)" and cannot be imported with type "{{.*}}" from the external library
// CHECK-NEXT: {{.*}}declaration_type_mismatch.c:10:{{.*}}: function declared here

extern float add_and_half_ptr(float a, float*);

void test(size_t size, float *input, float *output) {
  ripple_block_t BS = ripple_set_block_shape(VEC, 64);
  size_t BlockX = ripple_id(BS, 0);
  size_t BlockSizeX = ripple_get_block_size(BS, 0);
  size_t i;
  for (i = 0; i + BlockSizeX < size; i += BlockSizeX)
    output[i + BlockX] = add_and_half_ptr(input[i + BlockX], input);
}

// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -g -S -O2 -emit-llvm -fenable-ripple %s -o %t 2> %t2; FileCheck %s --input-file %t2

#include <ripple.h>
#define VEC_LANES 0

// CHECK: ripple does not allow implicit broadcasting of a store address to the value address; the value has shape 'Tensor[32]' and the address has shape 'Scalar'. Hint: use ripple_id() for the address computation or use a reduction operation
// CHECK-NEXT: 16 |         *scalar_output = values[i + vecIdx];
// CHECK-NEXT:    |                        ^

void test(size_t n, float *values, float *scalar_output) {
    ripple_block_t BS = ripple_set_block_shape(VEC_LANES, 32);
    size_t vecIdx = ripple_id(BS, 0);
    size_t vecSize = ripple_get_block_size(BS, 0);
    for (size_t i = 0; i < n; i += vecSize) {
        *scalar_output = values[i + vecIdx];
    }
}

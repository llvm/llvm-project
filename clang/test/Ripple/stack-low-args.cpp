// REQUIRES: target=hexagon{{.*}} || target-aarch64 || target-x86_64
// RUN: %clang -g -S -fenable-ripple -O0 -emit-llvm %s -o - 2>&1
// RUN: %clang -g -S -fenable-ripple -O1 -emit-llvm %s -o - 2>&1
// RUN: %clang -g -S -fenable-ripple -O2 -emit-llvm %s -o - 2>&1
// RUN: %clang -g -S -fenable-ripple -O3 -emit-llvm %s -o - 2>&1
// RUN: %clang -g -S -fenable-ripple -Os -emit-llvm %s -o - 2>&1
// RUN: %clang -g -S -fenable-ripple -Oz -emit-llvm %s -o - 2>&1

#include "ripple_test.h"
#define VEC_LANES 0

template <typename T>
void foo(const T (*base)[2][2][2], T *out) {
    ripple_block_t BS1 = ripple_set_block_shape(VEC_LANES, 2, 2, 2, 4);
    ripple_block_t BS2 = ripple_set_block_shape(VEC_LANES, 2, 2);

    size_t v0 = ripple_id(BS1, 0);
    size_t v1 = ripple_id(BS1, 1);
    size_t v2 = ripple_id(BS1, 2);
    size_t v3 = ripple_id(BS1, 3);

    T t0 = base[0][v2][v1][v0];
    T t1 = base[1][v2][v1][v0];
    T t2 = base[2][v2][v1][v0];
    T t3 = base[3][v2][v1][v0];

    out[v0 + 2 * v1] =
        ripple_stack(BS2, t0, t1, t2, t3);
}
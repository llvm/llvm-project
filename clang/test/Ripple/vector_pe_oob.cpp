// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -g -S -O2 -emit-llvm -fenable-ripple %s -o - 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"
#include <ripple.h>
#define HVX_LANES 0

template <typename T>
size_t argmax_row_flat(T * values, size_t n, T & max_val) {
    constexpr size_t n_lanes = 128 / sizeof(T);
    ripple_block_t BS = ripple_set_block_shape(HVX_LANES, n_lanes);
    size_t v  = ripple_id(BS, 0);
    size_t nv = ripple_get_block_size(BS, 0);
    T max_vec_val = ripple_broadcast(BS, 0x1, (T) 0);
    // Find the absolute max and argmax in max_vec_val & max_vec_base
    max_val = ripple_reducemax(0b1, max_vec_val);
    return max_val;
}


#define N 2048

void argmax_test(size_t * argmaxes) {
    uint16_t vals_u16[N];
    uint16_t max_u16;
    size_t argmax_u16 = argmax_row_flat(vals_u16, N, max_u16);
    argmaxes[0] = argmax_u16;
}

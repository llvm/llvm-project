// REQUIRES: hexagon-registered-target || aarch64-registered-target || x86-registered-target
// RUN: %clang -S %s -O2 -fenable-ripple -emit-llvm -Rpass=ripple -Rpass-missed=ripple -o - 2>&1 | FileCheck %s

#include<ripple.h>

void foo(int a[128][128], int b[128][128]) {
    ripple_block_t bs = ripple_set_block_shape(0, 128);
    size_t v0 = ripple_id(bs, 0);

    for (int i = 0; i < 64; i++) {
        // CHECK: remark: tensor access requires gather: base is a tensor with shape 128; cannot form contiguous loads
        int temp = a[v0][(v0 < 64) ? i : (64 - i)];

        // CHECK: remark: tensor access requires gather: non-constant stride along dimension 0 prevents coalesced vector load
        // CHECK: remark: tensor access requires scatter: strided memory access along dimension 0 prevents coalesced vector store
        a[v0][i + 63] = b[v0 * i][v0];

        // CHECK: remark: tensor access requires scatter: non-constant stride along dimension 0 prevents coalesced vector store
        b[v0 * i][v0] = temp;
    }
}
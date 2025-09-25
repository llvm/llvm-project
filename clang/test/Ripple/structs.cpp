// REQUIRES: hexagon-registered-target || aarch64-registered-target || x86-registered-target
// RUN: %clang -g -S -fenable-ripple -O0 -emit-llvm %s -o - 2>&1
// RUN: %clang -g -S -fenable-ripple -O1 -emit-llvm %s -o - 2>&1
// RUN: %clang -g -S -fenable-ripple -O2 -emit-llvm %s -o - 2>&1
// RUN: %clang -g -S -fenable-ripple -O3 -emit-llvm %s -o - 2>&1
// RUN: %clang -g -S -fenable-ripple -Os -emit-llvm %s -o - 2>&1
// RUN: %clang -g -S -fenable-ripple -Oz -emit-llvm %s -o - 2>&1

#include <ripple.h>

typedef struct double_double {
    double x;
    double y;
} dd;

void f(dd a[64][64], dd c[64][64]) {
    ripple_block_t bs = ripple_set_block_shape(0, 128);
    size_t v0 = ripple_id(bs, 0);

    for (int i = 0; i < 64; i++) {
        c[i][v0].y = a[63 - i][v0].y;
    }
}
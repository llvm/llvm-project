// RUN: %clang_nasan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// Test that provenance is correctly inherited through pointer arithmetic (GEP).

#include <stdio.h>

extern void __nasan_dump_state(void);

void process(int * __restrict a, int * __restrict b) {
    // Access through pointer arithmetic should inherit provenance
    int *a_offset = a + 1;
    int *b_offset = b + 1;

    *a_offset = 10;
    *b_offset = 20;
}

int main() {
    int x[3] = {0, 0, 0};
    int y[3] = {0, 0, 0};

    process(x, y);  // Valid: non-overlapping arrays

    __nasan_dump_state();
    printf("x[1] = %d, y[1] = %d\n", x[1], y[1]);
    return 0;
}

// CHECK: Violations detected: 0
// CHECK: x[1] = 10, y[1] = 20

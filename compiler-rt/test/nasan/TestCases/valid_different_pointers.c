// RUN: %clang_nasan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// Test that NASan does not report false positives when different pointers
// are passed to restrict-qualified parameters.

#include <stdio.h>

extern void __nasan_dump_state(void);

void write_both(int * __restrict a, int * __restrict b) {
    *a = 1;
    *b = 2;
}

int main() {
    int x = 0;
    int y = 0;
    write_both(&x, &y);  // Valid: different pointers
    __nasan_dump_state();
    printf("x = %d, y = %d\n", x, y);
    return 0;
}

// CHECK: Violations detected: 0
// CHECK: x = 1, y = 2

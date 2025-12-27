// RUN: %clang_nasan %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

// Test that NASan detects a noalias violation when the same pointer is passed
// to both restrict-qualified parameters.

void write_both(int * __restrict a, int * __restrict b) {
    *a = 1;
    *b = 2;  // This should be caught as a violation when a==b
}

int main() {
    int x = 0;
    write_both(&x, &x);  // Violates noalias: same pointer for both
    return 0;
}

// CHECK: NoAliasSanitizer: conflicting accesses via incompatible provenances
// CHECK: Memory address:
// CHECK: Accessing via provenances:
// CHECK: noalias parameter
// CHECK: Previously accessed via provenances:
// CHECK: noalias parameter

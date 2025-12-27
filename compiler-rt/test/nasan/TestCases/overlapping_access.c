// RUN: %clang_nasan %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

// Test that NASan detects overlapping memory access through noalias pointers.

void process_arrays(int * __restrict a, int * __restrict b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i * 2;  // Overlap violation when arrays overlap
    }
}

int main() {
    int arr[10];
    // Pass overlapping regions: a points to arr[0], b points to arr[2]
    // They overlap when both write to arr[2] through arr[4]
    process_arrays(&arr[0], &arr[2], 5);
    return 0;
}

// CHECK: NoAliasSanitizer: conflicting accesses via incompatible provenances

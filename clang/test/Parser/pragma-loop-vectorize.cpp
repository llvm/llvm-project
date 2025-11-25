// RUN: not %clang_cc1 -fsyntax-only %s 2>&1 | FileCheck %s

void sum_vector(unsigned int A[], unsigned int B[], unsigned int sum[]) {
    #pragma clang loop vectorize_width(4,8,16) vectorize(assume_safety)
    for (int k = 0; k < 64; k++) {
        sum[k] = A[k] + 3 * B[k];
    }
}

// CHECK: error: vectorize_width loop hint malformed; use
// CHECK-SAME: vectorize_width(X, fixed) or vectorize_width(X, scalable) where X is an integer, or
// CHECK-SAME: vectorize_width('fixed' or 'scalable')
// CHECK: warning: extra tokens at end of '#pragma clang loop vectorize_width'

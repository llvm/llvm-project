// RUN: %clang_cc1 -fsyntax-only %s -verify

void sum_vector(unsigned int A[], unsigned int B[], unsigned int sum[]) {
  #pragma clang loop vectorize_width(4,8,16) vectorize(assume_safety)
  // expected-error@-1 {{vectorize_width loop hint malformed; use vectorize_width(X, fixed) or vectorize_width(X, scalable) where X is an integer, or vectorize_width('fixed' or 'scalable')}}
  // expected-warning@-2 {{extra tokens at end of '#pragma clang loop vectorize_width' - ignored}}

  for (int k = 0; k < 64; k++) {
    sum[k] = A[k] + 3 * B[k];
  }
}

// RUN: %clang_cc1 -fopenmp -fsyntax-only -verify %s

void test_partial_const_var(int n) {
  const int factor = 4;
#pragma omp unroll partial(factor)
  for (int i = 0; i < n; ++i) {
  }
}

void test_partial_folded_exprs(int n) {
#pragma omp unroll partial((0, 4)) // expected-warning {{left operand of comma operator has no effect}}
  for (int i = 0; i < n; ++i) {
  }

#pragma omp unroll partial((int)(2.0 * 2.0))
  for (int i = 0; i < n; ++i) {
  }
}

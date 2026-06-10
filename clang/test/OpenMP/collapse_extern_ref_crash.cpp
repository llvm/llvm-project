// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=51 %s

// expected-no-diagnostics

// Verify there is no crash when collapsing a loop nest where the induction
// variable is an extern reference type.

extern int &dim;
auto test() {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < dim; ++i) {
    for (i = 0; i < 10; i++) {
      int dummy;
    }
  }
}

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=51 %s

// Verify no crash when collapsing a loop nest where the induction variable
// is an extern reference type. PR/issue: null dereference in getInitLCDecl
// when VarDecl::getDefinition() returns nullptr.

extern int &dim;
auto test() {
  // expected-error@+1 {{expected-error for malformed collapse}}
#pragma omp parallel for collapse(2)
  for (int i = 0; i < dim; ++i) {
    for (i = 0; i < 10; i++) {
      int dummy;
    }
  }
}

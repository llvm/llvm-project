// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fsyntax-only -verify %s

void test_valid(int **a) {
  #pragma omp target data map(to: a[0:1][0:100])
  {
    #pragma omp target update from(a[0:1][0:100])
  }
}

void test_missing_length(int **a) {
  #pragma omp target update from(a[0:1][0:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
  // expected-error@-1 {{expected at least one 'to' clause or 'from' clause specified}}
}
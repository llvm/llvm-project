// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 %s

void foo(int x, int n) {
  double vec[n];
  for (int iter = 0; iter < x; iter++) {
#pragma omp target teams distribute parallel for map( \
    from                                              \
    : vec [0:n]) default(none)
    // expected-error@+1 {{variable 'n' must have explicitly specified data sharing attributes}}
    for (int ii = 0; ii < n; ii++) {
      // expected-error@+3 {{variable 'iter' must have explicitly specified data sharing attributes}}
      // expected-error@+2 {{variable 'vec' must have explicitly specified data sharing attributes}}
      // expected-error@+1 {{variable 'x' must have explicitly specified data sharing attributes}}
      vec[ii] = iter + ii + x;
    }
  }
}


// RUN: %clang_cc1 -verify -fopenmp %s

// RUN: %clang_cc1 -verify -fopenmp-simd %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
void f(int a, ...) {
#pragma omp parallel for
  for (int i = 0; i < 100; ++i) {
    __builtin_va_list ap;
    __builtin_va_start(ap, a); // expected-error {{'va_start' cannot be used in a captured statement}}
  }
};

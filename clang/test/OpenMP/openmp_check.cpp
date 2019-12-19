// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s
// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -std=c++98 %s
// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -std=c++11 %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 -std=c++98 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 -std=c++11 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

#define p _Pragma("omp parallel")

int nested(int a) {
#pragma omp parallel p // expected-error {{unexpected OpenMP directive}}
  ++a;
#pragma omp parallel
  ++a;

  auto F = [&]() {
#if __cplusplus <= 199711L
  // expected-warning@-2 {{'auto' type specifier is a C++11 extension}}
  // expected-error@-3 {{expected expression}}
#endif

#pragma omp parallel
    {
#pragma omp target
      ++a;
    }
  };
  F();
  return a;
}

// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s
// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -std=c++98 %s
// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -std=c++11 %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 -std=c++98 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 -std=c++11 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

int nested(int a) {
#pragma omp parallel
  ++a;

  auto F = [&]() {
#if __cplusplus <= 199711L
  // expected-warning@-2 {{'auto' type specifier is a C++11 extension}}
  // expected-error@-3 {{expected expression}}
  // expected-error@-4 {{expected ';' at end of declaration}}
#else
  // expected-no-diagnostics
#endif

#pragma omp parallel
    {
#pragma omp target
      ++a;
    }
  };
  F();
#if __cplusplus <= 199711L
  // expected-error@-2 {{C++ requires a type specifier for all declarations}}
#endif
  return a;
#if __cplusplus <= 199711L
  // expected-error@-2 {{expected unqualified-id}}
#endif
}
#if __cplusplus <= 199711L
// expected-error@-2 {{extraneous closing brace ('}')}}
#endif

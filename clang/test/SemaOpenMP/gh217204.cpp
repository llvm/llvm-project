// RUN: %clang_cc1 -fopenmp -fsyntax-only -verify %s

namespace N {
  void foo();
}

#pragma omp declare simd // expected-error {{function declaration is expected after 'declare simd' directive}}
#pragma omp declare target to(N::foo)

#pragma omp declare variant // expected-error {{function declaration is expected after 'declare variant' directive}}
#pragma omp declare target to(N::foo)

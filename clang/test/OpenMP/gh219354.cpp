// RUN: %clang_cc1 -fopenmp -fsyntax-only -verify %s

template <class T, int N> &T::foo() { // expected-error {{nested name specifier 'T' for declaration does not refer into a class, class template or class template partial specialization}}
#pragma omp simd collapse(N)
  ;
}

template <class T, int N> void T::bar() { // expected-error {{nested name specifier 'T' for declaration does not refer into a class, class template or class template partial specialization}}
#pragma omp simd
  for (int i = 0; i < N; ++i)
    ;
#pragma omp taskloop collapse(N)
  for (int j = 0; j < 10; ++j)
    for (int k = 0; k < 10; ++k)
      ;
#pragma omp for ordered(N)
  for (int l = 0; l < 10; ++l)
    ;
}

template <int> struct E {
  void f();
};
template <int N> void E<0>::f() { // expected-error {{template parameter list matching the non-templated nested type 'E<0>' should be empty ('template<>')}}
#pragma omp simd collapse(N)
  for (int i = 0; i < 10; ++i)
    ;
}

// RUN: %clang_cc1 -std=c++20 -fopenmp -fblocks -fsyntax-only -verify %s

template <class T, int N> &T::foo() { // expected-error {{nested name specifier 'T' for declaration does not refer into a class, class template or class template partial specialization}}
#pragma omp simd collapse(N)
  ;
}

template <class T, int N> void T::bar() { // expected-error {{nested name specifier 'T' for declaration does not refer into a class, class template or class template partial specialization}}
  int lin = 0;
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
#pragma omp simd collapse(N) linear(lin)
  for (int m = 0; m < 10; ++m)
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

int lambda(auto x, int (*f)() = [] {
  int k = 0;
#pragma omp simd collapse(sizeof(x) / sizeof(x))
  for (int i = 0; i < 10; ++i)
    k++;
#pragma omp simd
  for (int i = 0; i < (int)sizeof(x); ++i)
    k++;
  return k;
}) {
  return f();
}
int useLambda() { return lambda(1); }

int lambdaErr(auto x, int (*f)() = [] { // expected-note {{while substituting into a lambda expression here}}
  int k = 0;
#pragma omp simd collapse(sizeof(x) / sizeof(x) + 1) // expected-note {{as specified in 'collapse' clause}}
  for (int i = 0; i < 10; ++i)
    k++; // expected-error {{expected 2 for loops after '#pragma omp simd', but found only 1}}
  return k;
}) {
  return f();
}
int useLambdaErr() { return lambdaErr(1); } // expected-note {{in instantiation of default function argument expression for 'lambdaErr<int>' required here}}

template <int N> void (^Block)() = ^{
#pragma omp simd collapse(N)
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      ;
#pragma omp parallel for
  for (int i = 0; i < N; ++i)
    ;
};
void useBlock() { Block<2>(); }

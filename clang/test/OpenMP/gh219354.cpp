// RUN: %clang_cc1 -fopenmp -fblocks -fsyntax-only -verify %s

template <class T, int N> &T::foo() { // expected-error {{nested name specifier 'T' for declaration does not refer into a class, class template or class template partial specialization}}
#pragma omp simd collapse(N)
  ;
}

template <class T, int N> void T::bar() { // expected-error {{nested name specifier 'T' for declaration does not refer into a class, class template or class template partial specialization}}
#pragma omp simd
  for (int i = 0; i < N; ++i)
    ;
#pragma omp simd
  for (auto x : T())
    ;
#pragma omp taskloop collapse(N)
  for (int j = 0; j < 10; ++j)
    for (int k = 0; k < 10; ++k)
      ;
#pragma omp for ordered(N)
  for (int l = 0; l < 10; ++l)
    ;
}

template <int N> void (^Block)() = ^{
  int a[N];
#pragma omp simd
  for (int x : a)
    ;
#pragma omp simd collapse(N)
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      ;
#pragma omp parallel for
  for (int i = 0; i < N; ++i)
    ;
};

void use() { Block<2>(); }

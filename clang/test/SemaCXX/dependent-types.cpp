// RUN: %clang_cc1 -fsyntax-only -pedantic -verify -std=c++11 %s

template<typename T> using U = int &;

template<typename T, int Size> void f() {
  T x1;
  T* x2;
  T& x3; // expected-error{{declaration of reference variable 'x3' requires an initializer}}
  T x4[]; // expected-error{{needs an explicit size or an initializer}}
  T x5[Size];
  int x6[Size];
  U<T> x7; // expected-error{{declaration of reference variable 'x7' requires an initializer}}
}

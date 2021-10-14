// RUN: %clang_cc1 %s -verify -fsyntax-only

void f1(int n) {
  typedef int x[n];
  const x y; // expected-error {{default initialization of an object of const type 'const x' (aka 'const int [n]')}}
}

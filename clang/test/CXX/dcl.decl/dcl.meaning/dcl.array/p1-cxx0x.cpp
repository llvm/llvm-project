// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11

void f() {
  int b[5];
  auto a[5] = b; // expected-error{{'a' declared as array of 'auto'}}
  auto *c[5] = b; // expected-error{{'c' declared as array of 'auto *'}}
}

// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11

void f() {
  int b[5];
  auto a[5] = b; // expected-error{{variable 'a' with type 'auto[5]' has incompatible initializer of type 'int[5]'}}
  auto *c[5] = b; // expected-error{{variable 'c' with type 'auto *[5]' has incompatible initializer of type 'int[5]'}}
}

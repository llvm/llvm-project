// RUN: %clang_cc1 %s -std=c++23 -fsyntax-only -verify

void f() {
  template for (char x : "123") {} // expected-error {{expansion statements are only supported in C++2c}}
}

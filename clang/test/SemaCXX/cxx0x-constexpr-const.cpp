// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

constexpr int x = 1;  // expected-note {{variable 'x' declared const here}}
constexpr int id(int x) { return x; }

void foo(void) {
  x = 2; // expected-error {{cannot assign to variable 'x' with const-qualified type 'const int'}}
  int (*idp)(int) = id;
}


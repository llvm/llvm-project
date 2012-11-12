// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

constexpr int x = 1;
constexpr int id(int x) { return x; }

void foo(void) {
  x = 2; // expected-error {{read-only variable is not assignable}}
  int (*idp)(int) = id;
}


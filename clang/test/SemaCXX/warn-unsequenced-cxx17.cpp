// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 -Wno-unused %s

void test() {
  int xs[10];
  int *p = xs;
  // expected-no-diagnostics
  p[(long long unsigned)(p = 0)]; // ok
}

// clang/test/Sema/implicit-deprecated-special-member.cpp
// RUN: %clang_cc1 -std=c++20 -Wdeprecated-declarations -verify %s

struct A {
  [[deprecated("use something else")]] int x = 42; // expected-note {{has been explicitly marked deprecated here}}
};

A makeDefaultA() {            // implicit default ctor: no diagnostics
  return {};
}

A copyA(const A &a) {         // implicit copy ctor: no diagnostics
  return a;
}

void assignA() {
  A a, b;
  a = b;                      // implicit copy-assign: no diagnostics
}

void useA() {
  A a;
  (void)a.x;                  // expected-warning {{is deprecated}}
}
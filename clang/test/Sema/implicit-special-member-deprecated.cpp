// RUN: %clang_cc1 -std=c++20 -Wdeprecated-declarations -verify %s

struct A {
  [[deprecated("use something else")]] int x = 42; // expected-note {{marked deprecated here}}
};

A makeDefaultA() { return {}; }    // ctor is implicit → no warn
A copyA(const A &a) { return a; }  // copy-ctor implicit → no warn

void assignA() {
  A a, b;
  a = b;                           // copy-assign implicit → no warn
}

void useA() {
  A a;
  (void)a.x;                       // expected-warning {{is deprecated}}
}

// Explicitly-defaulted ctor – now silent
struct B {
  [[deprecated]] int y;
  B() = default;                   // no warning under new policy
};

// RUN: %clang_cc1 -std=c++20 -Wdeprecated-declarations -I%S/Inputs -verify %s

#include "std-compare.h"

namespace GH147293 {
struct A {
  [[deprecated("use something else")]] int x = 42; // expected-note {{marked deprecated here}}
};

A makeDefaultA() { return {}; }    // ctor is implicit -> no warn
A copyA(const A &a) { return a; }  // copy-ctor implicit -> no warn

void assignA() {
  A a, b;
  a = b;                           // copy-assign implicit -> no warn
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

}

namespace GH147293_regression {

struct A {
  [[deprecated("use something else")]] int x = 42;
  auto operator<=>(const A&) const = default;
};

struct B : A {
  bool operator==(const B&) const = default;
};

void foo() {
  A x, y;
  (void)(x == y);
  (void)(x < y);

  B bx, by;
  (void)(bx != by);
}

}

namespace GH160543 {

template<class F>
struct [[deprecated]] X { X(F);}; // expected-warning {{is deprecated}} expected-note {{deprecated here}}

void f() {
  X x{0}; // expected-note {{while substituting}}
}

}


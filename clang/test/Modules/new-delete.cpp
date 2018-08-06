// RUN: %clang_cc1 -fmodules -verify %s
// expected-no-diagnostics

#pragma clang module build M
module M {}
#pragma clang module contents
#pragma clang module begin M
struct A {
  A();
  ~A() { delete p; } // expected-warning {{'delete' applied to a pointer that was allocated with 'new[]'}}
  int *p;
};
inline A::A() : p(new int[32]) {} // expected-note {{allocated}}
struct B {
  B();
  ~B() { delete p; }
  int *p;
};
#pragma clang module end
#pragma clang module endbuild

#pragma clang module import M
B::B() : p(new int[32]) {}

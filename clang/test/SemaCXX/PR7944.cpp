// RUN: %clang_cc1 -fsyntax-only -verify %s
// PR7944

#define MACRO(x) x

struct B { int f() { return 0; } };
struct A { B* b() { return new B; } };

void g() {
  A a;
  MACRO(a.b->f());  // expected-error{{reference to non-static member function must be called; did you mean to call it with no arguments?}}
}

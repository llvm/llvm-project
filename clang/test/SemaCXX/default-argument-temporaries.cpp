// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
struct B { B(void* = 0); };

struct A {
  A(B b = B()) { }
};

void f() {
  (void)B();
  (void)A();
}

// RUN: %clang_cc1 -fsyntax-only -fobjc-arc -verify -fblocks %s
// expected-no-diagnostics

struct X0 {
  static id makeObject1() __attribute__((ns_returns_retained));
  id makeObject2() __attribute__((ns_returns_retained));
};

void test_X0(X0 x0, X0 *x0p) {
  X0::makeObject1();
  x0.makeObject2();
  x0p->makeObject2();
  id (X0::*pmf)() __attribute__((ns_returns_retained)) = &X0::makeObject2;
  (x0.*pmf)();
  (x0p->*pmf)();
}

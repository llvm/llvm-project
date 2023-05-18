// RUN: %clang_cc1 -emit-llvm-only -verify %s
// expected-no-diagnostics

struct A { int a(); };
typedef int B;
void a() {
  A x;
  ((x.a))();
  ((x.*&A::a))();
  B y;
  // FIXME: Sema doesn't like this for some reason...
  //(y.~B)();
}

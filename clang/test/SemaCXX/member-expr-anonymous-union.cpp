// RUN: %clang_cc1 %s -fsyntax-only -verify
// expected-no-diagnostics
// PR5543

struct A { int x; union { int* y; float* z; }; }; struct B : A {int a;};
int* a(B* x) { return x->y; }

struct x { union { int y; }; }; x y; template <int X> int f() { return X+y.y; }
int g() { return f<2>(); }


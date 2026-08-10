// RUN: %clang_cc1 %s -fno-rtti -emit-llvm-only -o - -triple=i386-pc-win32 -verify

// A is not trivially copyable and must be passed indirectly or with inalloca.
struct A {
  A();
  A(const A &o);
  virtual ~A();
  int a;
};

struct B {
  B();
  int b;
  virtual B *clone(A);
};

// Converting from C* to B* requires a this adjustment.
struct C : A, B {
  C();
  int c;
  virtual C *clone(A); // expected-error {{cannot compile this non-trivial argument copy for return-adjusting thunk yet}}
};
B::B() {}  // force emission
C::C() {}  // force emission

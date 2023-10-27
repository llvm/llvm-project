// RUN: %clang_cc1 -fsyntax-only -verify %s

struct B1 {
  void f();
  static void f(int);
  int i; // expected-note 2{{member found by ambiguous name lookup}}
};
struct B2 {
  void f(double);
};
struct I1: B1 { };
struct I2: B1 { };

struct D: I1, I2, B2 {
  using B1::f;
  using B2::f;
  void g() {
    f(); // expected-error {{ambiguous conversion from derived class 'D' to base class 'B1'}}
    f(0); // ok
    f(0.0); // ok
    // FIXME next line should be well-formed
    int B1::* mpB1 = &D::i; // expected-error {{non-static member 'i' found in multiple base-class subobjects of type 'B1'}}
    int D::* mpD = &D::i; // expected-error {{non-static member 'i' found in multiple base-class subobjects of type 'B1'}}
  }
};

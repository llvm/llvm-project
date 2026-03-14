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

namespace GH80435 {
struct A {
  void *data; // expected-note {{member found by ambiguous name lookup}}
};

class B {
  void *data; // expected-note {{member found by ambiguous name lookup}}
};

struct C : A, B {};

decltype(C().data) x; // expected-error {{member 'data' found in multiple base classes of different types}}

struct D { // expected-note {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'C' to 'const D' for 1st argument}}
           // expected-note@-1{{candidate constructor (the implicit move constructor) not viable: no known conversion from 'C' to 'D' for 1st argument}}
  template <typename Container, decltype(Container().data) = 0 >
  D(Container); // expected-note {{candidate template ignored: substitution failure [with Container = C]: member 'data' found in multiple base classes of different types}}
};

D y(C{}); // expected-error {{no matching constructor for initialization of 'D'}}
}

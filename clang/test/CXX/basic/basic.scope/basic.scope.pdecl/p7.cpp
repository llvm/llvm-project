// RUN: %clang_cc1 -verify %s

// Some of the diagnostics produce 'did you mean?' suggestions. We don't care
// which ones for the purpose of this test.
// expected-note@* 0+{{here}}

struct A *f();
using Test_A = A;

void f(struct B*);
using Test_B = B;

struct C;
using Test_C = C;

struct X {
  struct D *f();
  void f(struct E*);
  struct F;
  friend struct G;
};
using Test_D = D;
using Test_XD = X::D; // expected-error {{no type named 'D' in 'X'}}
using Test_E = E;
using Test_XE = X::E; // expected-error {{no type named 'E' in 'X'}}
using Test_F = F; // expected-error {{unknown type name 'F'}}
using Test_XF = X::F;
using Test_G = G; // expected-error {{unknown type name 'G'}}
using Test_XG = X::G; // expected-error {{no type named 'G' in 'X'}}

void g() {
  {
    struct X {
      struct H *f();
      void f(struct I*);
      struct J;
      friend struct K;
    };
    using Test_H = H;
    using Test_XH = X::H; // expected-error {{no type named}}
    using Test_I = I;
    using Test_XI = X::I; // expected-error {{no type named}}
    using Test_J = J; // expected-error {{unknown type name}}
    using Test_XJ = X::J;
    using Test_K = K; // expected-error {{unknown type name}}
    using Test_XK = X::K; // expected-error {{no type named}}
  }
  using Test_H = H; // expected-error {{unknown type name}}
  using Test_I = I; // expected-error {{unknown type name}}
  using Test_J = J; // expected-error {{unknown type name}}
  using Test_K = K; // expected-error {{unknown type name}}
}
using Test_H = H; // expected-error {{unknown type name}}
using Test_I = I; // expected-error {{unknown type name}}
using Test_J = J; // expected-error {{unknown type name}}
using Test_K = K; // expected-error {{unknown type name}}

// RUN: %clang_cc1 -std=c++11 -verify %s
//
// Note: [class.inhctor] was removed by P0136R1. This tests the new behavior
// for the wording that used to be there.

struct A { // expected-note 8{{candidate is the implicit}}
  A(...); // expected-note 4{{candidate constructor}} expected-note 4{{candidate inherited constructor}}
  A(int = 0, int = 0, int = 0, int = 0, ...); // expected-note 3{{candidate constructor}} expected-note 3{{candidate inherited constructor}}
  A(int = 0, int = 0, ...); // expected-note 3{{candidate constructor}} expected-note 3{{candidate inherited constructor}}

  template<typename T> A(T, int = 0, ...); // expected-note 3{{candidate constructor}} expected-note 3{{candidate inherited constructor}}

  template<typename T, int N> A(const T (&)[N]); // expected-note {{candidate constructor}} expected-note {{candidate inherited constructor}}
  template<typename T, int N> A(const T (&)[N], int = 0); // expected-note {{candidate constructor}} expected-note {{candidate inherited constructor}}
};

struct B : A { // expected-note 4{{candidate is the implicit}}
  using A::A; // expected-note 19{{inherited here}}
  B(void*);
};

struct C {} c;

A a0{}; // expected-error {{ambiguous}}
B b0{}; // expected-error {{ambiguous}}

A a1{1}; // expected-error {{ambiguous}}
B b1{1}; // expected-error {{ambiguous}}

A a2{1,2}; // expected-error {{ambiguous}}
B b2{1,2}; // expected-error {{ambiguous}}

A a3{1,2,3}; // ok
B b3{1,2,3}; // ok

A a4{1,2,3,4}; // ok
B b4{1,2,3,4}; // ok

A a5{1,2,3,4,5}; // ok
B b5{1,2,3,4,5}; // ok

A a6{c}; // ok
B b6{c}; // ok

A a7{c,0}; // ok
B b7{c,0}; // ok

A a8{c,0,1}; // ok
B b8{c,0,1}; // ok

A a9{"foo"}; // expected-error {{ambiguous}}
B b9{"foo"}; // expected-error {{ambiguous}}

namespace PR15755 {
  struct X {
    template<typename...Ts> X(int, Ts...);
  };
  struct Y : X {
    using X::X;
  };
  struct Z : Y {
    using Y::Y;
  };
  Z z(0);
}

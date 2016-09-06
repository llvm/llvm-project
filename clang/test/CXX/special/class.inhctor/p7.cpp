// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
//
// Note: [class.inhctor] was removed by P0136R1. This tests the new behavior
// for the wording that used to be there.

struct B1 { // expected-note 2{{candidate}}
  B1(int); // expected-note {{candidate}}
};
struct B2 { // expected-note 2{{candidate}}
  B2(int); // expected-note {{candidate}}
};
struct D1 : B1, B2 { // expected-note 2{{candidate}}
  using B1::B1; // expected-note 3{{inherited here}}
  using B2::B2; // expected-note 3{{inherited here}}
};
struct D2 : B1, B2 {
  using B1::B1;
  using B2::B2;
  D2(int);
};
D1 d1(0); // expected-error {{ambiguous}}
D2 d2(0);

template<typename T> struct B3 {
  B3(T);
};
template<typename T> struct B4 : B3<T>, B1 {
  B4();
  using B3<T>::B3;
  using B1::B1;
};
B4<char> b4c;
B4<int> b4i;

struct B5 {
  template<typename T> B5(T);
};
struct D6 : B5 {
  using B5::B5;
  template<typename T> D6(T);
};
D6 d6(0);
struct D7 : B5 {
  using B5::B5;
  template<typename T> D7(T, ...);
};
// DRxxx (no number yet): derived class ctor beats base class ctor.
D7 d7(0);

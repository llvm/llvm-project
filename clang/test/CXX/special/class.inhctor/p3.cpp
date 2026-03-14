// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
//
// Note: [class.inhctor] was removed by P0136R1. This tests the new behavior
// for the wording that used to be there.

struct B1 {
  B1(int); // expected-note 3{{target of using}}
  B1(int, int); // expected-note 3{{target of using}}
};
struct D1 : B1 {
  using B1::B1;
};
D1 d1a(1), d1b(1, 1);

D1 fd1() { return 1; }

struct B2 {
  explicit B2(int, int = 0, int = 0); // expected-note {{not a candidate}}
};
struct D2 : B2 { // expected-note 2{{candidate constructor}}
  using B2::B2;
};
D2 d2a(1), d2b(1, 1), d2c(1, 1, 1);

D2 fd2() { return 1; } // expected-error {{no viable conversion}}

struct B3 {
  B3(void*); // expected-note {{candidate}}
};
struct D3 : B3 { // expected-note 2{{candidate constructor}}
  using B3::B3; // expected-note {{inherited here}}
};
D3 fd3() { return 1; } // expected-error {{no viable conversion}}

template<typename T> struct T1 : B1 {
  using B1::B1; // expected-note 2{{using declaration}}
};
template<typename T> struct T2 : T1<T> {
  using T1<int>::T1; // expected-note 2{{using declaration}}
};
template<typename T> struct T3 : T1<int> {
  using T1<T>::T1; // expected-note 2{{using declaration}}
};
struct U {
  // [dcl.meaning]p1: "the member shall not merely hav ebeen introduced by a
  // using-declaration in the scope of the class [...] nominated by the
  // nested-name-specifier of the declarator-id"
  friend T1<int>::T1(int); // expected-error {{cannot befriend target of using declaration}}
  friend T1<int>::T1(int, int); // expected-error {{cannot befriend target of using declaration}}
  friend T2<int>::T2(int); // expected-error {{cannot befriend target of using declaration}}
  friend T2<int>::T2(int, int); // expected-error {{cannot befriend target of using declaration}}
  friend T3<int>::T3(int); // expected-error {{cannot befriend target of using declaration}}
  friend T3<int>::T3(int, int); // expected-error {{cannot befriend target of using declaration}}
};

struct B4 {
  template<typename T> explicit B4(T, int = 0); // expected-note 2{{here}}
};
template<typename T> struct T4 : B4 {
  using B4::B4;
  template<typename U> T4(U);
};
template<typename T> struct U4 : T4<T> {
  using T4<T>::T4;
};
T4<void> t4a = {0};
T4<void> t4b = {0, 0}; // expected-error {{chosen constructor is explicit}}
U4<void> u4a = {0};
U4<void> u4b = {0, 0}; // expected-error {{chosen constructor is explicit}}

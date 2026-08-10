// RUN: %clang_cc1 -verify %s -std=c++14

template<const int I> struct S { // expected-error {{reference member 'r' binds to a temporary object}}
  decltype(I) n;
  int &&r = I; // expected-note {{default member initializer}}
};
S<5> s; // expected-note {{implicit default constructor}}

template<typename T, T v> struct U { // expected-error {{reference member 'r' binds to a temporary object}}
  decltype(v) n;
  int &&r = v; // expected-note {{default member initializer}}
};
U<const int, 6> u; // expected-note {{implicit default constructor}}

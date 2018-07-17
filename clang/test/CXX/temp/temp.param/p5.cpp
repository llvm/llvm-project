// RUN: %clang_cc1 -verify %s -std=c++14

template<const int I> struct S { // expected-note {{in default member initializer}}
  decltype(I) n;
  int &&r = I; // expected-error {{reference member 'r' binds to a temporary object}}
};
S<5> s; // expected-note {{implicit default constructor}}

template<typename T, T v> struct U { // expected-note {{in default member initializer}}
  decltype(v) n;
  int &&r = v; // expected-error {{reference member 'r' binds to a temporary object}}
};
U<const int, 6> u; // expected-note {{implicit default constructor}}

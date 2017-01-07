// RUN: %clang_cc1 -verify %s -std=c++14

template<const int I> struct S {
  decltype(I) n;
  int &&r = I; // expected-warning 2{{binding reference member 'r' to a temporary value}} expected-note 2{{declared here}}
};
S<5> s; // expected-note {{instantiation}}

template<typename T, T v> struct U {
  decltype(v) n;
  int &&r = v; // expected-warning {{binding reference member 'r' to a temporary value}} expected-note {{declared here}}
};
U<const int, 6> u; // expected-note {{instantiation}}

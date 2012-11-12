// RUN: %clang_cc1 -verify %s -std=c++11
// expected-no-diagnostics

template<const int I> struct S {
  decltype(I) n;
  int &&r = I;
};
S<5> s;

template<typename T, T v> struct U {
  decltype(v) n;
  int &&r = v;
};
U<const int, 6> u;

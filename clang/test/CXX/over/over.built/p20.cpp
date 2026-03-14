// RUN: %clang_cc1 -std=c++11 -verify %s -Wno-tautological-compare

template <typename T, typename U>
void f(int* pi, float* pf, T* pt, U* pu, T t) {
  pi = pi;
  pi = pf; // expected-error {{incompatible pointer types}}
  pi = pt;
  pu = pi;
  pu = pt;
  pi = t;
  pu = t;
}

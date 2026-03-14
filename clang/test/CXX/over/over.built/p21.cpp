// RUN: %clang_cc1 -std=c++11 -verify %s -Wno-tautological-compare

struct A {
};

template <typename T>
void f(int A::* pi, float A::* pf, int T::* pt, T A::* pu, T t) {
  pi = pi;
  pi = pf; // expected-error {{assigning to 'int A::*' from incompatible type 'float A::*'}}
  pi = pt;
  pi = pu;
  pi = t;
}

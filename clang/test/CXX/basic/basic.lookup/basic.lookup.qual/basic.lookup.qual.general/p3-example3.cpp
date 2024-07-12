// RUN: %clang_cc1 -std=c++23 %s -verify

int f();

struct A {
  int B, C; // expected-note {{declared as a non-template here}}
  template<int> using D = void;
  using T = void;
  void f();
};

using B = A;
template<int> using C = A;
template<int> using D = A;
template<int> using X = A;

template<class T>
void g(T *p) {
  p->X<0>::f(); // expected-error {{no member named 'X' in 'A'}}
  p->template X<0>::f();
  p->B::f();
  p->template C<0>::f(); // expected-error {{'C' following the 'template' keyword does not refer to a template}}
  p->template D<0>::f(); // expected-error {{type 'template D<0>' (aka 'void') cannot be used prior to '::' because it has no members}}
  p->T::f(); // expected-error {{'A::T' (aka 'void') is not a class, namespace, or enumeration}}
}

template void g(A*); // expected-note {{in instantiation of}}

// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

template<typename T>
struct A : T {
  using T::f;
  using T::g;
  using T::h;

  void f();
  void g();

  void i() {
    f<int>();
    g<int>(); // expected-error{{no member named 'g' in 'A<B>'}}
    h<int>(); // expected-error{{expected '(' for function-style cast or type construction}}
              // expected-error@-1{{expected expression}}
  }
};

struct B {
  template<typename T>
  void f();

  void g();

  template<typename T>
  void h();
};

template struct A<B>; // expected-note{{in instantiation of member function 'A<B>::i' requested here}}

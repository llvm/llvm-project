// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

template<typename T>
struct A : T {
  using T::f;
  using T::g;

  void f();
  void g();

  void h() {
    f<int>();
    g<int>(); // expected-error{{no member named 'g' in 'A<B>'}}
  }
};

struct B {
  template<typename T>
  void f();

  void g();
};

template struct A<B>; // expected-note{{in instantiation of member function 'A<B>::h' requested here}}

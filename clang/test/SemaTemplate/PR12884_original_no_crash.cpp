// RUN: %clang_cc1 -verify -std=c++20 -fsyntax-only %s

namespace PR12884_original {
  template <typename T> struct A {
    struct B {
      template <typename U> struct X {};
      typedef int arg;
    };
    struct C {
      typedef B::X<typename B::arg> x; // expected-error{{typename specifier refers to non-type member 'arg' in 'PR12884_original::A<int>::B'}}
    };
  };

  template <> struct A<int>::B {
    template <int N> struct X {};
    static const int arg = 0; // expected-note{{referenced member 'arg' is declared here}}
  };

  A<int>::C::x a; // expected-note{{in instantiation of member class 'PR12884_original::A<int>::C' requested here}}
}

// RUN: %clang_cc1 -fsyntax-only -verify %s
namespace N {
  template<typename T> class A { };

  template<> class A<int> { };

  template<> class A<float>; // expected-note{{forward declaration of 'N::A<float>'}}

  class B : public A<int> { };
}

class C1 : public N::A<int> { };

class C2 : public N::A<float> { }; // expected-error{{base class has incomplete type}}

struct D1 {
  operator N::A<int>();
};

namespace N {
  struct D2 {
    operator A<int>();
  };
}

// RUN: %clang_cc1 -std=c++23 -Wno-unused %s -verify

namespace Unambiguous {
  struct A {
    int x;

    template<typename T>
    using C = A;
  };

  using B = A;

  template<typename T>
  using D = A;

  using E = void;

  struct F : A {
    void non_template() {
      this->x;
      this->A::x;
      this->B::x;
      this->C<int>::x;
      this->D<int>::x;
      this->E::x; // expected-error {{'Unambiguous::E' (aka 'void') is not a class, namespace, or enumeration}}
    }
  };

  template<typename T>
  void not_instantiated(T t) {
    t.x;
    t.A::x;
    t.B::x;
    t.C<int>::x; // expected-warning {{use 'template' keyword to treat 'C' as a dependent template name}}
    t.template C<int>::x;
    t.D<int>::x; // expected-warning {{use 'template' keyword to treat 'D' as a dependent template name}}
    t.template D<int>::x;
    t.E::x;
  }

  template<typename T>
  void instantiated_valid(T t) {
    t.x;
    t.A::x;
    t.B::x;
    t.template C<int>::x;
    t.template D<int>::x;
    t.E::x;
  }

  template<typename T>
  void instantiated_invalid(T t) {
    t.x;
    t.A::x;
    t.B::x; // expected-error {{'Unambiguous::Invalid::B' (aka 'void') is not a class, namespace, or enumeration}}
    t.template C<int>::x;
    t.template D<int>::x; // expected-error {{'D' following the 'template' keyword does not refer to a template}}
    t.E::x; // expected-error {{'Unambiguous::E' (aka 'void') is not a class, namespace, or enumeration}}
  }

  struct Valid : A {
    using E = A;
  };

  template void instantiated_valid(Valid);

  struct Invalid : A {
    using B = void;
    using D = A; // expected-note {{declared as a non-template here}}
  };

  template void instantiated_invalid(Invalid); // expected-note {{in instantiation of}}
} // namespace Unambiguous

namespace Ambiguous {
  inline namespace N {
    struct A { }; // expected-note {{candidate found by name lookup is 'Ambiguous::N::A'}}
  }

  struct A { }; // expected-note {{candidate found by name lookup is 'Ambiguous::A'}}

  template<typename T>
  void f(T t) {
    t.A::x; // expected-error {{reference to 'A' is ambiguous}}
  }

  struct B {
    using A = B;

    int x;
  };

  struct C { };

  template void f(B);
  template void f(C); // expected-note {{in instantiation of}}

} // namespace Ambiguous

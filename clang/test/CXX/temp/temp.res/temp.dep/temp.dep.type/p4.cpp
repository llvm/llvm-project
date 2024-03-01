// RUN: %clang_cc1 -Wno-unused-value -verify %s

namespace N0 {
  template<typename T>
  struct A {
    int x;
    void f();
    using X = int;

    void not_instantiated(A *a, A &b) {
      x;
      f();
      new X;

      this->x;
      this->f();
      this->A::x;
      this->A::f();

      a->x;
      a->f();
      a->A::x;
      a->A::f();

      (*this).x;
      (*this).f();
      (*this).A::x;
      (*this).A::f();

      b.x;
      b.f();
      b.A::x;
      b.A::f();

      A::x;
      A::f();
      new A::X;

      y; // expected-error{{use of undeclared identifier 'y'}}
      g(); // expected-error{{use of undeclared identifier 'g'}}
      new Y; // expected-error{{unknown type name 'Y'}}

      this->y; // expected-error{{no member named 'y' in 'A<T>'}}
      this->g(); // expected-error{{no member named 'g' in 'A<T>'}}
      this->A::y; // expected-error{{no member named 'y' in 'A<T>'}}
      this->A::g(); // expected-error{{no member named 'g' in 'A<T>'}}

      a->y; // expected-error{{no member named 'y' in 'A<T>'}}
      a->g(); // expected-error{{no member named 'g' in 'A<T>'}}
      a->A::y; // expected-error{{no member named 'y' in 'A<T>'}}
      a->A::g(); // expected-error{{no member named 'g' in 'A<T>'}}

      // FIXME: An overloaded unary 'operator*' is built for these
      // even though the operand is a pointer (to a dependent type).
      // Type::isOverloadableType should return false for such cases.
      (*this).y;
      (*this).g();
      (*this).A::y;
      (*this).A::g();

      b.y; // expected-error{{no member named 'y' in 'A<T>'}}
      b.g(); // expected-error{{no member named 'g' in 'A<T>'}}
      b.A::y; // expected-error{{no member named 'y' in 'A<T>'}}
      b.A::g(); // expected-error{{no member named 'g' in 'A<T>'}}

      A::y; // expected-error{{no member named 'y' in 'A<T>'}}
      A::g(); // expected-error{{no member named 'g' in 'A<T>'}}
      new A::Y; // expected-error{{no type named 'Y' in 'A<T>'}}
    }
  };
} // namespace N0

namespace N1 {
  struct A {
    template<int I>
    void f();
  };

  template<typename T>
  struct B {
    template<int I>
    void f();

    A x;
    A g();

    void not_instantiated(B *a, B &b) {
      f<0>();
      this->f<0>();
      a->f<0>();
      // FIXME: This should not require 'template'!
      (*this).f<0>(); // expected-error{{missing 'template' keyword prior to dependent template name 'f'}}
      b.f<0>();

      x.f<0>();
      this->x.f<0>();
      a->x.f<0>();
      // FIXME: This should not require 'template'!
      (*this).x.f<0>(); // expected-error{{missing 'template' keyword prior to dependent template name 'f'}}
      b.x.f<0>();

      // FIXME: None of these should require 'template'!
      g().f<0>(); // expected-error{{missing 'template' keyword prior to dependent template name 'f'}}
      this->g().f<0>(); // expected-error{{missing 'template' keyword prior to dependent template name 'f'}}
      a->g().f<0>(); // expected-error{{missing 'template' keyword prior to dependent template name 'f'}}
      (*this).g().f<0>(); // expected-error{{missing 'template' keyword prior to dependent template name 'f'}}
      b.g().f<0>(); // expected-error{{missing 'template' keyword prior to dependent template name 'f'}}
    }
  };
} // namespace N1

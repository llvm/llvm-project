// RUN: %clang_cc1 -Wno-unused-value -verify %s

namespace N0 {
  struct A {
    int x0;
    static int x1;
    int x2;
    static int x3;

    void f0();
    static void f1();
    void f2();
    static void f3();

    using M0 = int;
    using M1 = int;

    struct C0 { };
    struct C1 { };
  };

  template<typename T>
  struct B : A {
    int x4;
    static int x5;

    using A::x2;
    using A::x3;

    void f4();
    static void f5();

    using A::f2;
    using A::f3;

    using M2 = int;

    using A::M1;

    struct C2 { };

    using A::C1;

    void not_instantiated(B *a, B &b) {
      // All of the following should be found in the current instantiation.

      new M0;
      new B::M0;
      new A::M0;
      new B::A::M0;
      new C0;
      new B::C0;
      new A::C0;
      new B::A::C0;
      new M1;
      new B::M1;
      new A::M1;
      new B::A::M1;
      new C1;
      new B::C1;
      new A::C1;
      new B::A::C1;
      new M2;
      new B::M2;
      new C2;
      new B::C2;

      x0;
      B::x0;
      A::x0;
      B::A::x0;
      x1;
      B::x1;
      A::x1;
      B::A::x1;
      x2;
      B::x2;
      A::x2;
      B::A::x2;
      x3;
      B::x3;
      A::x3;
      B::A::x3;
      x4;
      B::x4;
      x5;
      B::x5;

      f0();
      B::f0();
      A::f0();
      B::A::f0();
      f1();
      B::f1();
      A::f1();
      B::A::f1();
      f2();
      B::f2();
      A::f2();
      B::A::f2();
      f3();
      B::f3();
      A::f3();
      B::A::f3();
      f4();
      B::f4();
      f5();
      B::f5();

      this->x0;
      this->B::x0;
      this->A::x0;
      this->B::A::x0;
      this->x1;
      this->B::x1;
      this->A::x1;
      this->B::A::x1;
      this->x2;
      this->B::x2;
      this->A::x2;
      this->B::A::x2;
      this->x3;
      this->B::x3;
      this->A::x3;
      this->B::A::x3;
      this->x4;
      this->B::x4;
      this->x5;
      this->B::x5;

      this->f0();
      this->B::f0();
      this->A::f0();
      this->B::A::f0();
      this->f1();
      this->B::f1();
      this->A::f1();
      this->B::A::f1();
      this->f2();
      this->B::f2();
      this->A::f2();
      this->B::A::f2();
      this->f3();
      this->B::f3();
      this->A::f3();
      this->B::A::f3();
      this->f4();
      this->B::f4();
      this->f5();
      this->B::f5();

      a->x0;
      a->B::x0;
      a->A::x0;
      a->B::A::x0;
      a->x1;
      a->B::x1;
      a->A::x1;
      a->B::A::x1;
      a->x2;
      a->B::x2;
      a->A::x2;
      a->B::A::x2;
      a->x3;
      a->B::x3;
      a->A::x3;
      a->B::A::x3;
      a->x4;
      a->B::x4;
      a->x5;
      a->B::x5;

      a->f0();
      a->B::f0();
      a->A::f0();
      a->B::A::f0();
      a->f1();
      a->B::f1();
      a->A::f1();
      a->B::A::f1();
      a->f2();
      a->B::f2();
      a->A::f2();
      a->B::A::f2();
      a->f3();
      a->B::f3();
      a->A::f3();
      a->B::A::f3();
      a->f4();
      a->B::f4();
      a->f5();
      a->B::f5();

      (*this).x0;
      (*this).B::x0;
      (*this).A::x0;
      (*this).B::A::x0;
      (*this).x1;
      (*this).B::x1;
      (*this).A::x1;
      (*this).B::A::x1;
      (*this).x2;
      (*this).B::x2;
      (*this).A::x2;
      (*this).B::A::x2;
      (*this).x3;
      (*this).B::x3;
      (*this).A::x3;
      (*this).B::A::x3;
      (*this).x4;
      (*this).B::x4;
      (*this).x5;
      (*this).B::x5;

      (*this).f0();
      (*this).B::f0();
      (*this).A::f0();
      (*this).B::A::f0();
      (*this).f1();
      (*this).B::f1();
      (*this).A::f1();
      (*this).B::A::f1();
      (*this).f2();
      (*this).B::f2();
      (*this).A::f2();
      (*this).B::A::f2();
      (*this).f3();
      (*this).B::f3();
      (*this).A::f3();
      (*this).B::A::f3();
      (*this).f4();
      (*this).B::f4();
      (*this).f5();
      (*this).B::f5();

      b.x0;
      b.B::x0;
      b.A::x0;
      b.B::A::x0;
      b.x1;
      b.B::x1;
      b.A::x1;
      b.B::A::x1;
      b.x2;
      b.B::x2;
      b.A::x2;
      b.B::A::x2;
      b.x3;
      b.B::x3;
      b.A::x3;
      b.B::A::x3;
      b.x4;
      b.B::x4;
      b.x5;
      b.B::x5;

      b.f0();
      b.B::f0();
      b.A::f0();
      b.B::A::f0();
      b.f1();
      b.B::f1();
      b.A::f1();
      b.B::A::f1();
      b.f2();
      b.B::f2();
      b.A::f2();
      b.B::A::f2();
      b.f3();
      b.B::f3();
      b.A::f3();
      b.B::A::f3();
      b.f4();
      b.B::f4();
      b.f5();
      b.B::f5();

      // None of the following should be found in the current instantiation.

      new M3; // expected-error{{unknown type name 'M3'}}
      new B::M3; // expected-error{{no type named 'M3' in 'B<T>'}}
      new A::M3; // expected-error{{no type named 'M3' in 'N0::A'}}
      new B::A::M3; // expected-error{{no type named 'M3' in 'N0::A'}}

      x6; // expected-error{{use of undeclared identifier 'x6'}}
      B::x6; // expected-error{{no member named 'x6' in 'B<T>'}}
      A::x6; // expected-error{{no member named 'x6' in 'N0::A'}}
      B::A::x6; // expected-error{{no member named 'x6' in 'N0::A'}}
      f6(); // expected-error{{use of undeclared identifier 'f6'}}
      B::f6(); // expected-error{{no member named 'f6' in 'B<T>'}}
      A::f6(); // expected-error{{no member named 'f6' in 'N0::A'}}
      B::A::f6(); // expected-error{{no member named 'f6' in 'N0::A'}}

      this->x6; // expected-error{{no member named 'x6' in 'B<T>'}}
      this->B::x6; // expected-error{{no member named 'x6' in 'B<T>'}}
      this->A::x6; // expected-error{{no member named 'x6' in 'N0::A'}}
      this->B::A::x6; // expected-error{{no member named 'x6' in 'N0::A'}}
      this->f6(); // expected-error{{no member named 'f6' in 'B<T>'}}
      this->B::f6(); // expected-error{{no member named 'f6' in 'B<T>'}}
      this->A::f6(); // expected-error{{no member named 'f6' in 'N0::A'}}
      this->B::A::f6(); // expected-error{{no member named 'f6' in 'N0::A'}}

      a->x6; // expected-error{{no member named 'x6' in 'B<T>'}}
      a->B::x6; // expected-error{{no member named 'x6' in 'B<T>'}}
      a->A::x6; // expected-error{{no member named 'x6' in 'N0::A'}}
      a->B::A::x6; // expected-error{{no member named 'x6' in 'N0::A'}}
      a->f6(); // expected-error{{no member named 'f6' in 'B<T>'}}
      a->B::f6(); // expected-error{{no member named 'f6' in 'B<T>'}}
      a->A::f6(); // expected-error{{no member named 'f6' in 'N0::A'}}
      a->B::A::f6(); // expected-error{{no member named 'f6' in 'N0::A'}}

      // FIXME: An overloaded unary 'operator*' is built for these
      // even though the operand is a pointer (to a dependent type).
      // Type::isOverloadableType should return false for such cases.
      (*this).x6;
      (*this).B::x6;
      (*this).A::x6;
      (*this).B::A::x6;
      (*this).f6();
      (*this).B::f6();
      (*this).A::f6();
      (*this).B::A::f6();

      b.x6; // expected-error{{no member named 'x6' in 'B<T>'}}
      b.B::x6; // expected-error{{no member named 'x6' in 'B<T>'}}
      b.A::x6; // expected-error{{no member named 'x6' in 'N0::A'}}
      b.B::A::x6; // expected-error{{no member named 'x6' in 'N0::A'}}
      b.f6(); // expected-error{{no member named 'f6' in 'B<T>'}}
      b.B::f6(); // expected-error{{no member named 'f6' in 'B<T>'}}
      b.A::f6(); // expected-error{{no member named 'f6' in 'N0::A'}}
      b.B::A::f6(); // expected-error{{no member named 'f6' in 'N0::A'}}
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

// RUN: %clang_cc1 -Wno-unused-value -verify %s

namespace N0 {
  struct A {
    int x0;
    static int y0;
    int x1;
    static int y1;

    void f0();
    static void g0();
    void f1();
    static void g1();

    using M0 = int;
    using M1 = int;

    struct C0 { };
    struct C1 { };
  };

  template<typename T>
  struct B : A {
    int x2;
    static int y2;

    void f2();
    static void g2();

    using M2 = int;

    struct C2 { };

    using A::x1;
    using A::y1;
    using A::f1;
    using A::g1;
    using A::M1;
    using A::C1;

    using T::x3;
    using T::y3;
    using T::f3;
    using T::g3;
    using typename T::M3;
    using typename T::C3;

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
      new M3;
      new B::M3;
      new C3;
      new B::C3;

      x0;
      B::x0;
      A::x0;
      B::A::x0;
      y0;
      B::y0;
      A::y0;
      B::A::y0;
      x1;
      B::x1;
      A::x1;
      B::A::x1;
      y1;
      B::y1;
      A::y1;
      B::A::y1;
      x2;
      B::x2;
      y2;
      B::y2;
      x3;
      B::x3;
      y3;
      B::y3;

      f0();
      B::f0();
      A::f0();
      B::A::f0();
      g0();
      B::g0();
      A::g0();
      B::A::g0();
      f1();
      B::f1();
      A::f1();
      B::A::f1();
      g1();
      B::g1();
      A::g1();
      B::A::g1();
      f2();
      B::f2();
      g2();
      B::g2();
      f3();
      B::f3();
      g3();
      B::g3();

      this->x0;
      this->B::x0;
      this->A::x0;
      this->B::A::x0;
      this->y0;
      this->B::y0;
      this->A::y0;
      this->B::A::y0;
      this->x1;
      this->B::x1;
      this->A::x1;
      this->B::A::x1;
      this->y1;
      this->B::y1;
      this->A::y1;
      this->B::A::y1;
      this->x2;
      this->B::x2;
      this->y2;
      this->B::y2;
      this->x3;
      this->B::x3;
      this->y3;
      this->B::y3;

      this->f0();
      this->B::f0();
      this->A::f0();
      this->B::A::f0();
      this->g0();
      this->B::g0();
      this->A::g0();
      this->B::A::g0();
      this->f1();
      this->B::f1();
      this->A::f1();
      this->B::A::f1();
      this->g1();
      this->B::g1();
      this->A::g1();
      this->B::A::g1();
      this->f2();
      this->B::f2();
      this->g2();
      this->B::g2();
      this->f3();
      this->B::f3();
      this->g3();
      this->B::g3();

      a->x0;
      a->B::x0;
      a->A::x0;
      a->B::A::x0;
      a->y0;
      a->B::y0;
      a->A::y0;
      a->B::A::y0;
      a->x1;
      a->B::x1;
      a->A::x1;
      a->B::A::x1;
      a->y1;
      a->B::y1;
      a->A::y1;
      a->B::A::y1;
      a->x2;
      a->B::x2;
      a->y2;
      a->B::y2;
      a->x3;
      a->B::x3;
      a->y3;
      a->B::y3;

      a->f0();
      a->B::f0();
      a->A::f0();
      a->B::A::f0();
      a->g0();
      a->B::g0();
      a->A::g0();
      a->B::A::g0();
      a->f1();
      a->B::f1();
      a->A::f1();
      a->B::A::f1();
      a->g1();
      a->B::g1();
      a->A::g1();
      a->B::A::g1();
      a->f2();
      a->B::f2();
      a->g2();
      a->B::g2();
      a->f3();
      a->B::f3();
      a->g3();
      a->B::g3();

      (*this).x0;
      (*this).B::x0;
      (*this).A::x0;
      (*this).B::A::x0;
      (*this).y0;
      (*this).B::y0;
      (*this).A::y0;
      (*this).B::A::y0;
      (*this).x1;
      (*this).B::x1;
      (*this).A::x1;
      (*this).B::A::x1;
      (*this).y1;
      (*this).B::y1;
      (*this).A::y1;
      (*this).B::A::y1;
      (*this).x2;
      (*this).B::x2;
      (*this).y2;
      (*this).B::y2;
      (*this).x3;
      (*this).B::x3;
      (*this).y3;
      (*this).B::y3;

      (*this).f0();
      (*this).B::f0();
      (*this).A::f0();
      (*this).B::A::f0();
      (*this).g0();
      (*this).B::g0();
      (*this).A::g0();
      (*this).B::A::g0();
      (*this).f1();
      (*this).B::f1();
      (*this).A::f1();
      (*this).B::A::f1();
      (*this).g1();
      (*this).B::g1();
      (*this).A::g1();
      (*this).B::A::g1();
      (*this).f2();
      (*this).B::f2();
      (*this).g2();
      (*this).B::g2();
      (*this).f3();
      (*this).B::f3();
      (*this).g3();
      (*this).B::g3();

      b.x0;
      b.B::x0;
      b.A::x0;
      b.B::A::x0;
      b.y0;
      b.B::y0;
      b.A::y0;
      b.B::A::y0;
      b.x1;
      b.B::x1;
      b.A::x1;
      b.B::A::x1;
      b.y1;
      b.B::y1;
      b.A::y1;
      b.B::A::y1;
      b.x2;
      b.B::x2;
      b.y2;
      b.B::y2;
      b.x3;
      b.B::x3;
      b.y3;
      b.B::y3;

      b.f0();
      b.B::f0();
      b.A::f0();
      b.B::A::f0();
      b.g0();
      b.B::g0();
      b.A::g0();
      b.B::A::g0();
      b.f1();
      b.B::f1();
      b.A::f1();
      b.B::A::f1();
      b.g1();
      b.B::g1();
      b.A::g1();
      b.B::A::g1();
      b.f2();
      b.B::f2();
      b.g2();
      b.B::g2();
      b.f3();
      b.B::f3();
      b.g3();
      b.B::g3();

      // None of the following should be found in the current instantiation.

      new M4; // expected-error{{unknown type name 'M4'}}
      new B::M4; // expected-error{{no type named 'M4' in 'B<T>'}}
      new A::M4; // expected-error{{no type named 'M4' in 'N0::A'}}
      new B::A::M4; // expected-error{{no type named 'M4' in 'N0::A'}}

      x4; // expected-error{{use of undeclared identifier 'x4'}}
      B::x4; // expected-error{{no member named 'x4' in 'B<T>'}}
      A::x4; // expected-error{{no member named 'x4' in 'N0::A'}}
      B::A::x4; // expected-error{{no member named 'x4' in 'N0::A'}}
      f4(); // expected-error{{use of undeclared identifier 'f4'}}
      B::f4(); // expected-error{{no member named 'f4' in 'B<T>'}}
      A::f4(); // expected-error{{no member named 'f4' in 'N0::A'}}
      B::A::f4(); // expected-error{{no member named 'f4' in 'N0::A'}}

      this->x4; // expected-error{{no member named 'x4' in 'B<T>'}}
      this->B::x4; // expected-error{{no member named 'x4' in 'B<T>'}}
      this->A::x4; // expected-error{{no member named 'x4' in 'N0::A'}}
      this->B::A::x4; // expected-error{{no member named 'x4' in 'N0::A'}}
      this->f4(); // expected-error{{no member named 'f4' in 'B<T>'}}
      this->B::f4(); // expected-error{{no member named 'f4' in 'B<T>'}}
      this->A::f4(); // expected-error{{no member named 'f4' in 'N0::A'}}
      this->B::A::f4(); // expected-error{{no member named 'f4' in 'N0::A'}}

      a->x4; // expected-error{{no member named 'x4' in 'B<T>'}}
      a->B::x4; // expected-error{{no member named 'x4' in 'B<T>'}}
      a->A::x4; // expected-error{{no member named 'x4' in 'N0::A'}}
      a->B::A::x4; // expected-error{{no member named 'x4' in 'N0::A'}}
      a->f4(); // expected-error{{no member named 'f4' in 'B<T>'}}
      a->B::f4(); // expected-error{{no member named 'f4' in 'B<T>'}}
      a->A::f4(); // expected-error{{no member named 'f4' in 'N0::A'}}
      a->B::A::f4(); // expected-error{{no member named 'f4' in 'N0::A'}}

      (*this).x4; // expected-error{{no member named 'x4' in 'B<T>'}}
      (*this).B::x4; // expected-error{{no member named 'x4' in 'B<T>'}}
      (*this).A::x4; // expected-error{{no member named 'x4' in 'N0::A'}}
      (*this).B::A::x4; // expected-error{{no member named 'x4' in 'N0::A'}}
      (*this).f4(); // expected-error{{no member named 'f4' in 'B<T>'}}
      (*this).B::f4(); // expected-error{{no member named 'f4' in 'B<T>'}}
      (*this).A::f4(); // expected-error{{no member named 'f4' in 'N0::A'}}
      (*this).B::A::f4(); // expected-error{{no member named 'f4' in 'N0::A'}}

      b.x4; // expected-error{{no member named 'x4' in 'B<T>'}}
      b.B::x4; // expected-error{{no member named 'x4' in 'B<T>'}}
      b.A::x4; // expected-error{{no member named 'x4' in 'N0::A'}}
      b.B::A::x4; // expected-error{{no member named 'x4' in 'N0::A'}}
      b.f4(); // expected-error{{no member named 'f4' in 'B<T>'}}
      b.B::f4(); // expected-error{{no member named 'f4' in 'B<T>'}}
      b.A::f4(); // expected-error{{no member named 'f4' in 'N0::A'}}
      b.B::A::f4(); // expected-error{{no member named 'f4' in 'N0::A'}}
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
      (*this).f<0>();
      b.f<0>();

      x.f<0>();
      this->x.f<0>();
      a->x.f<0>();
      (*this).x.f<0>();
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

namespace N2 {
  template<typename T>
  struct A {
    struct B {
      using C = A;

      void not_instantiated(A *a, B *b) {
        b->x; // expected-error{{no member named 'x' in 'N2::A::B'}}
        b->B::x; // expected-error{{no member named 'x' in 'N2::A::B'}}
        a->B::C::x; // expected-error{{no member named 'x' in 'A<T>'}}
      }
    };

    void not_instantiated(A *a, B *b) {
      b->x;
      b->B::x;
      a->B::C::x;
    }
  };
} // namespace N2

namespace N3 {
  struct A { };

  template<typename T>
  struct B : A {
    void not_instantiated() {
      // Dependent, lookup context is the current instantiation.
      this->operator=(*this);
      // Not dependent, the lookup context is A (not the current instantiation).
      this->A::operator=(*this);
    }
  };

  template<typename T>
  struct C {
    template<typename U>
    void operator=(int);

    void not_instantiated() {
      operator=<int>(0);
      C::operator=<int>(0);
      this->operator=<int>(0);
      this->C::operator=<int>(0);

      operator=(*this);
      C::operator=(*this);
      this->operator=(*this);
      this->C::operator=(*this);
    }
  };

  template<typename T>
  struct D {
    auto not_instantiated() -> decltype(operator=(0)); // expected-error {{use of undeclared 'operator='}}
  };

  template<typename T>
  struct E {
    auto instantiated(E& e) -> decltype(operator=(e)); // expected-error {{use of undeclared 'operator='}}
  };

  template struct E<int>; // expected-note {{in instantiation of template class 'N3::E<int>' requested here}}
} // namespace N3

namespace N4 {
  template<typename T>
  struct A {
    void not_instantiated(A a, A<T> b, T c) {
      a->x; // expected-error {{member reference type 'A<T>' is not a pointer; did you mean to use '.'?}}
      b->x; // expected-error {{member reference type 'A<T>' is not a pointer; did you mean to use '.'?}}
      c->x;
    }

    void instantiated(A a, A<T> b, T c) {
      // FIXME: We should only emit a single diagnostic suggesting to use '.'!
      a->x; // expected-error {{member reference type 'A<T>' is not a pointer; did you mean to use '.'?}}
            // expected-error@-1 {{member reference type 'A<int>' is not a pointer; did you mean to use '.'?}}
            // expected-error@-2 {{no member named 'x' in 'N4::A<int>'}}
      b->x; // expected-error {{member reference type 'A<T>' is not a pointer; did you mean to use '.'?}}
            // expected-error@-1 {{member reference type 'A<int>' is not a pointer; did you mean to use '.'?}}
            // expected-error@-2 {{no member named 'x' in 'N4::A<int>'}}
      c->x; // expected-error {{member reference type 'int' is not a pointer}}
    }
  };

  template void A<int>::instantiated(A<int>, A<int>, int); // expected-note {{in instantiation of}}

  struct B {
    int x;

    void f();
  };

  template<typename T>
  struct C {
    B *operator->();

    void not_instantiated(C a, C<T> b, T c) {
      a->x;
      b->x;
      c->x;
    }

    void instantiated(C a, C<T> b, T c) {
      a->x;
      b->x;
      c->x; // expected-error {{member reference type 'int' is not a pointer}}
    }
  };

  template void C<int>::instantiated(C, C, int); // expected-note {{in instantiation of}}

  template<typename T>
  struct D {
    T *operator->();

    void not_instantiated(D a) {
      a->x;
      a->y;
      a->f();
      a->g();

      a->T::x;
      a->T::y;
      a->T::f();
      a->T::g();

      a->U::x;
      a->U::y;
      a->U::f();
      a->U::g();
    }

    void instantiated(D a) {
      a->x;
      a->y; // expected-error {{no member named 'y' in 'N4::B'}}
      a->f();
      a->g(); // expected-error {{no member named 'g' in 'N4::B'}}

      a->T::x;
      a->T::y; // expected-error {{no member named 'y' in 'N4::B'}}
      a->T::f();
      a->T::g(); // expected-error {{no member named 'g' in 'N4::B'}}
    }
  };

  template void D<B>::instantiated(D); // expected-note {{in instantiation of}}

  template<typename T>
  struct Typo {
    T *operator->();

    void not_instantiated(Typo a) {
      a->Not_instantiated;
      a->typo;
      a->T::Not_instantiated;
      a->T::typo;
    }
  };
} // namespace N4

namespace N5 {
  struct A {
    int x;
  };

  template<typename T>
  void f() {
    A y = T::x; // expected-error {{type 'int' cannot be used prior to '::' because it has no members}}
    y.x;
  }

  template void f<int>(); // expected-note {{in instantiation of}}

  struct B {
    template<typename T>
    B(T&&);

    int x;
  };

  template<typename T>
  void g(T y) {
    B z([&]() { // expected-note {{while substituting into a lambda expression here}}
      h(&y); // expected-error {{use of undeclared identifier 'h'}}
    });
    z.x;
  }

  template void g(int); // expected-note {{in instantiation of}}
} // namespace N5

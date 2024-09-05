// RUN: %clang_cc1 -fms-extensions -fsyntax-only -Wno-unused-value -verify %s
// RUN: %clang_cc1 -fms-extensions -fdelayed-template-parsing -fsyntax-only -Wno-unused-value -verify %s

class A {
public:
  template<class U> A(U p) {}
  template<> A(int p) {}

  template<class U> void f(U p) {}

  template<> void f(int p) {}

  void f(int p) {}
};

void test1() {
  A a(3);
  char *b;
  a.f(b);
  a.f<int>(99);
  a.f(100);
}

template<class T> class B {
public:
  template<class U> B(U p) {}
  template<> B(int p) {}

  template<class U> void f(U p) { T y = 9; }

  template<> void f(int p) {
    T a = 3;
  }

  void f(int p) { T a = 3; }
};

void test2() {
  B<char> b(3);
  char *ptr;
  b.f(ptr);
  b.f<int>(99);
  b.f(100);
}

namespace PR12709 {
  template<class T> class TemplateClass {
    void member_function() { specialized_member_template<false>(); }

    template<bool b> void specialized_member_template() {}

    template<> void specialized_member_template<false>() {}
  };

  void f() { TemplateClass<int> t; }
}

namespace Duplicates {
  template<typename T> struct A {
    template<typename U> void f();
    template<> void f<int>() {}
    template<> void f<T>() {}
  };

  // FIXME: We should diagnose the duplicate explicit specialization definitions
  // here.
  template struct A<int>;
}

namespace PR28082 {
struct S {
  template <int>
  int f(int = 0);
  template <>
  int f<0>(int);
};
}

namespace UsesThis {
  template<typename T>
  struct A {
    int x;

    static inline int y;

    template<typename U = void>
    static void f();

    template<typename U = void>
    void g();

    template<typename U>
    static auto h() -> A*;

    void i();

    static void j();

    template<>
    void f<int>() {
      this->x; // expected-error {{invalid use of 'this' outside of a non-static member function}}
      x; // expected-error {{invalid use of member 'x' in static member function}}
      A::x; // expected-error {{invalid use of member 'x' in static member function}}
      +x; // expected-error {{invalid use of member 'x' in static member function}}
      +A::x; // expected-error {{invalid use of member 'x' in static member function}}
      &x; // expected-error {{invalid use of member 'x' in static member function}}
      &A::x;
      this->y; // expected-error {{invalid use of 'this' outside of a non-static member function}}
      y;
      A::y;
      +y;
      +A::y;
      &y;
      &A::y;
      f();
      f<void>();
      g(); // expected-error {{call to non-static member function without an object argument}}
      g<void>(); // expected-error {{call to non-static member function without an object argument}}
      i(); // expected-error {{call to non-static member function without an object argument}}
      j();
      &i; // expected-error 2{{must explicitly qualify name of member function when taking its address}}
      &j;
      &A::i;
      &A::j;
    }

    template<>
    void g<int>() {
      this->x;
      x;
      A::x;
      +x;
      +A::x;
      &x;
      &A::x;
      this->y;
      y;
      A::y;
      +y;
      +A::y;
      &y;
      &A::y;
      f();
      f<void>();
      g();
      g<void>();
      i();
      j();
      &i; // expected-error 2{{must explicitly qualify name of member function when taking its address}}
      &j;
      &A::i;
      &A::j;
    }

    template<>
    auto h<int>() -> decltype(this); // expected-error {{'this' cannot be used in a static member function declaration}}
  };

  template struct A<int>; // expected-note 3{{in instantiation of}}

  template <typename T>
  struct Foo {
    template <typename X>
    int bar(X x) {
      return 0;
    }

    template <>
    int bar(int x) {
      return bar(5.0); // ok
    }
  };

  void call() {
    Foo<double> f;
    f.bar(1);
  }

  struct B {
    int x0;
    static inline int y0;

    int f0(int);
    static int g0(int);

    int x2;
    static inline int y2;

    int f2(int);
    static int g2(int);
  };

  template<typename T>
  struct D : B {
    int x1;
    static inline int y1;

    int f1(int);
    static int g1(int);

    using B::x2;
    using B::y2;
    using B::f2;
    using B::g2;

    template<typename U>
    void non_static_spec(U);

    template<typename U>
    static void static_spec(U);

    template<>
    void non_static_spec(int z) {
      ++z;
      ++x0;
      ++x1;
      ++x2;
      ++y0;
      ++y1;
      ++y2;

      &z;
      &x0;
      &x1;
      &x2;
      &y0;
      &y1;
      &y2;

      &f0; // expected-error {{must explicitly qualify name of member function when taking its address}}
      &f1; // expected-error 2{{must explicitly qualify name of member function when taking its address}}
      &f2; // expected-error 2{{must explicitly qualify name of member function when taking its address}}
      &g0;
      &g1;
      &g2;

      &B::x0;
      &D::x1;
      &B::x2;
      &B::y0;
      &D::y1;
      &B::y2;
      &B::f0;
      &D::f1;
      &B::f2;
      &B::g0;
      &D::g1;
      &B::g2;

      f0(0);
      f0(z);
      f0(x0);
      f0(x1);
      f0(x2);
      f0(y0);
      f0(y1);
      f0(y2);
      g0(0);
      g0(z);
      g0(x0);
      g0(x1);
      g0(x2);
      g0(y0);
      g0(y1);
      g0(y2);

      f1(0);
      f1(z);
      f1(x0);
      f1(x1);
      f1(x2);
      f1(y0);
      f1(y1);
      f1(y2);
      g1(0);
      g1(z);
      g1(x0);
      g1(x1);
      g1(x2);
      g1(y0);
      g1(y1);
      g1(y2);

      f2(0);
      f2(z);
      f2(x0);
      f2(x1);
      f2(x2);
      f2(y0);
      f2(y1);
      f2(y2);
      g2(0);
      g2(z);
      g2(x0);
      g2(x1);
      g2(x2);
      g2(y0);
      g2(y1);
      g2(y2);
    }

    template<>
    void static_spec(int z) {
      ++z;
      ++x0; // expected-error {{invalid use of member 'x0' in static member function}}
      ++x1; // expected-error {{invalid use of member 'x1' in static member function}}
      ++x2; // expected-error {{invalid use of member 'x2' in static member function}}
      ++y0;
      ++y1;
      ++y2;

      &z;
      &x0; // expected-error {{invalid use of member 'x0' in static member function}}
      &x1; // expected-error {{invalid use of member 'x1' in static member function}}
      &x2; // expected-error {{invalid use of member 'x2' in static member function}}
      &y0;
      &y1;
      &y2;

      &f0; // expected-error {{must explicitly qualify name of member function when taking its address}}
      &f1; // expected-error 2{{must explicitly qualify name of member function when taking its address}}
      &f2; // expected-error 2{{must explicitly qualify name of member function when taking its address}}
      &g0;
      &g1;
      &g2;

      &B::x0;
      &D::x1;
      &B::x2;
      &B::y0;
      &D::y1;
      &B::y2;
      &B::f0;
      &D::f1;
      &B::f2;
      &B::g0;
      &D::g1;
      &B::g2;

      f0(0); // expected-error {{call to non-static member function without an object argument}}
      f0(z); // expected-error {{call to non-static member function without an object argument}}
      f0(x0); // expected-error {{call to non-static member function without an object argument}}
      f0(x1); // expected-error {{call to non-static member function without an object argument}}
      f0(x2); // expected-error {{call to non-static member function without an object argument}}
      f0(y0); // expected-error {{call to non-static member function without an object argument}}
      f0(y1); // expected-error {{call to non-static member function without an object argument}}
      f0(y2); // expected-error {{call to non-static member function without an object argument}}
      g0(0);
      g0(z);
      g0(x0); // expected-error {{invalid use of member 'x0' in static member function}}
      g0(x1); // expected-error {{invalid use of member 'x1' in static member function}}
      g0(x2); // expected-error {{invalid use of member 'x2' in static member function}}
      g0(y0);
      g0(y1);
      g0(y2);

      f1(0); // expected-error {{call to non-static member function without an object argument}}
      f1(z); // expected-error {{call to non-static member function without an object argument}}
      f1(x0); // expected-error {{call to non-static member function without an object argument}}
      f1(x1); // expected-error {{call to non-static member function without an object argument}}
      f1(x2); // expected-error {{call to non-static member function without an object argument}}
      f1(y0); // expected-error {{call to non-static member function without an object argument}}
      f1(y1); // expected-error {{call to non-static member function without an object argument}}
      f1(y2); // expected-error {{call to non-static member function without an object argument}}
      g1(0);
      g1(z);
      g1(x0); // expected-error {{invalid use of member 'x0' in static member function}}
      g1(x1); // expected-error {{invalid use of member 'x1' in static member function}}
      g1(x2); // expected-error {{invalid use of member 'x2' in static member function}}
      g1(y0);
      g1(y1);
      g1(y2);

      f2(0); // expected-error {{call to non-static member function without an object argument}}
      f2(z); // expected-error {{call to non-static member function without an object argument}}
      f2(x0); // expected-error {{call to non-static member function without an object argument}}
      f2(x1); // expected-error {{call to non-static member function without an object argument}}
      f2(x2); // expected-error {{call to non-static member function without an object argument}}
      f2(y0); // expected-error {{call to non-static member function without an object argument}}
      f2(y1); // expected-error {{call to non-static member function without an object argument}}
      f2(y2); // expected-error {{call to non-static member function without an object argument}}
      g2(0);
      g2(z);
      g2(x0); // expected-error {{invalid use of member 'x0' in static member function}}
      g2(x1); // expected-error {{invalid use of member 'x1' in static member function}}
      g2(x2); // expected-error {{invalid use of member 'x2' in static member function}}
      g2(y0);
      g2(y1);
      g2(y2);
    }
  };

  template struct D<int>; // expected-note 2{{in instantiation of}}

  template<typename T>
  struct E : T {
    int x1;
    static inline int y1;

    int f1(int);
    static int g1(int);

    using T::x0;
    using T::y0;
    using T::f0;
    using T::g0;

    template<typename U>
    void non_static_spec(U);

    template<typename U>
    static void static_spec(U);

    template<>
    void non_static_spec(int z) {
      ++z;
      ++x0;
      ++x1;
      ++y0;
      ++y1;

      &z;
      &x0;
      &x1;
      &y0;
      &y1;

      &f0; // expected-error {{must explicitly qualify name of member function when taking its address}}
      &f1; // expected-error 2{{must explicitly qualify name of member function when taking its address}}
      &g0;
      &g1;

      &T::x0;
      &E::x1;
      &T::y0;
      &E::y1;
      &T::f0;
      &E::f1;
      &T::g0;
      &E::g1;

      f0(0);
      f0(z);
      f0(x0);
      f0(x1);
      f0(y0);
      f0(y1);
      g0(0);
      g0(z);
      g0(x0);
      g0(x1);
      g0(y0);
      g0(y1);

      f1(0);
      f1(z);
      f1(x0);
      f1(x1);
      f1(y0);
      f1(y1);
      g1(0);
      g1(z);
      g1(x0);
      g1(x1);
      g1(y0);
      g1(y1);

      T::f0(0);
      T::f0(z);
      T::f0(x0);
      T::f0(x1);
      T::f0(y0);
      T::f0(y1);
      T::g0(0);
      T::g0(z);
      T::g0(x0);
      T::g0(x1);
      T::g0(y0);
      T::g0(y1);

      E::f1(0);
      E::f1(z);
      E::f1(x0);
      E::f1(x1);
      E::f1(y0);
      E::f1(y1);
      E::g1(0);
      E::g1(z);
      E::g1(x0);
      E::g1(x1);
      E::g1(y0);
      E::g1(y1);
    }

    template<>
    void static_spec(int z) {
      ++z;
      ++x0; // expected-error {{invalid use of member 'x0' in static member function}}
      ++x1; // expected-error {{invalid use of member 'x1' in static member function}}
      ++y0;
      ++y1;

      &z;
      &x0; // expected-error {{invalid use of member 'x0' in static member function}}
      &x1; // expected-error {{invalid use of member 'x1' in static member function}}
      &y0;
      &y1;

      &f0; // expected-error {{must explicitly qualify name of member function when taking its address}}
      &f1; // expected-error 2{{must explicitly qualify name of member function when taking its address}}
      &g0;
      &g1;

      &T::x0;
      &E::x1;
      &T::y0;
      &E::y1;
      &T::f0;
      &E::f1;
      &T::g0;
      &E::g1;

      f0(0); // expected-error {{call to non-static member function without an object argument}}
      f0(z); // expected-error {{call to non-static member function without an object argument}}
      f0(x0); // expected-error {{call to non-static member function without an object argument}}
      f0(x1); // expected-error {{call to non-static member function without an object argument}}
      f0(y0); // expected-error {{call to non-static member function without an object argument}}
      f0(y1); // expected-error {{call to non-static member function without an object argument}}
      g0(0);
      g0(z);
      g0(x0); // expected-error {{invalid use of member 'x0' in static member function}}
      g0(x1); // expected-error {{invalid use of member 'x1' in static member function}}
      g0(y0);
      g0(y1);

      f1(0); // expected-error {{call to non-static member function without an object argument}}
      f1(z); // expected-error {{call to non-static member function without an object argument}}
      f1(x0); // expected-error {{call to non-static member function without an object argument}}
      f1(x1); // expected-error {{call to non-static member function without an object argument}}
      f1(y0); // expected-error {{call to non-static member function without an object argument}}
      f1(y1); // expected-error {{call to non-static member function without an object argument}}
      g1(0);
      g1(z);
      g1(x0); // expected-error {{invalid use of member 'x0' in static member function}}
      g1(x1); // expected-error {{invalid use of member 'x1' in static member function}}
      g1(y0);
      g1(y1);

      T::f0(0); // expected-error {{call to non-static member function without an object argument}}
      T::f0(z); // expected-error {{call to non-static member function without an object argument}}
      T::f0(x0); // expected-error {{call to non-static member function without an object argument}}
      T::f0(x1); // expected-error {{call to non-static member function without an object argument}}
      T::f0(y0); // expected-error {{call to non-static member function without an object argument}}
      T::f0(y1); // expected-error {{call to non-static member function without an object argument}}
      T::g0(0);
      T::g0(z);
      T::g0(x0); // expected-error {{invalid use of member 'x0' in static member function}}
      T::g0(x1); // expected-error {{invalid use of member 'x1' in static member function}}
      T::g0(y0);
      T::g0(y1);

      E::f1(0); // expected-error {{call to non-static member function without an object argument}}
      E::f1(z); // expected-error {{call to non-static member function without an object argument}}
      E::f1(x0); // expected-error {{call to non-static member function without an object argument}}
      E::f1(x1); // expected-error {{call to non-static member function without an object argument}}
      E::f1(y0); // expected-error {{call to non-static member function without an object argument}}
      E::f1(y1); // expected-error {{call to non-static member function without an object argument}}
      E::g1(0);
      E::g1(z);
      E::g1(x0); // expected-error {{invalid use of member 'x0' in static member function}}
      E::g1(x1); // expected-error {{invalid use of member 'x1' in static member function}}
      E::g1(y0);
      E::g1(y1);
    }
  };

  template struct E<B>; // expected-note 2{{in instantiation of}}

}

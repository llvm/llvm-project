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
}

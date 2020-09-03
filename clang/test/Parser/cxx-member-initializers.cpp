// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

struct x {
  x() : a(4) ; // expected-error {{expected '{'}}
};

struct y {
  int a;
  y() : a(4) ; // expected-error {{expected '{'}}
};

struct z {
  int a;
  z() : a {}
}; // expected-error {{expected '{'}}

namespace PR16480 {
  template<int n> struct X {
    X();
    X(int);
  };

  struct A : X<0> {
    A() : X<a<b>{0}.n>() {}

    template<int> struct a {
      int n;
    };

    static const int b = 1;
  };

  struct B : X<0> {
    B() : X<a<b>{0} {}

    static const int a = 0, b = 0;
  };

  template<int> struct a {
    constexpr a(int) {}
    constexpr operator int() const { return 0; }
  };

  struct C : X<0> {
    C() : X<a<b>(0)>() {}

    static const int b = 0;
  };

  struct D : X<0> {
    D() : X<a<b>(0) {}

    static const int a = 0, b = 0;
  };

  template<typename T> struct E : X<0> {
    E(X<0>) : X<(0)>{} {}
    E(X<1>) : X<int{}>{} {}
    E(X<2>) : X<(0)>() {}
    E(X<3>) : X<int{}>() {}
  };

  // FIXME: This should be valid in the union of C99 and C++11.
  struct F : X<0> {
    F() : X<A<T>().n + (T){}.n>{} {} // expected-error +{{}} expected-note {{to match}}

    struct T { int n; };
    template<typename> struct A { int n; };
  };

  // FIXME: This is valid now, but may be made ill-formed by DR1607.
  struct G : X<0> {
    G() : X<0 && [](){return 0;}()>{} // expected-error +{{}} expected-note {{to match}}
  };

  struct Errs : X<0> {
    Errs(X<0>) : decltype X<0>() {} // expected-error {{expected '(' after 'decltype'}}
    Errs(X<1>) : what is this () {} // expected-error {{expected '(' or '{'}}
    Errs(X<2>) : decltype(X<0> // expected-note {{to match this '('}}
  }; // expected-error {{expected ')'}}
}

template <class U, class V> struct C {
  int f() { return 4; }
  class C1 {};
};

class D {};
namespace N {
struct E {
  class F {};
};
}

class G {
  // These are all valid:
  void f(int x = C<int, D>().f()) {}
  void g(int x = C<int, ::D>().f()) {}
  void h(int x = C<int, N::E>().f()) {}
  void i(int x = C<int, ::N::E>().f()) {}
  void j(int x = C<int, decltype(N::E())::F>().f()) {}
  void k(int x = C<int, C<int, int>>().f()) {}
  void l(int x = C<int, C<int, int>::C1>().f()) {}

  // This isn't, but it shouldn't crash. The diagnostics don't matter much.
  void m(int x = C<int, union int>().f()) {} // expected-error {{declaration of anonymous union must be a definition}} expected-error {{expected a type}} expected-error {{expected '>'}} expected-note {{to match}}
};

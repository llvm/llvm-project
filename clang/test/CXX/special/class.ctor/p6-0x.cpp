// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11

// Implicitly-defined default constructors are constexpr if the implicit
// definition would be.
struct NonConstexpr1 { // expected-note {{here}}
  int a;
};
struct NonConstexpr2 { // expected-note {{here}}
  NonConstexpr1 nl;
};
struct NonConstexpr2a : NonConstexpr1 { };
constexpr NonConstexpr1 nc1 = NonConstexpr1(); // ok, does not call constructor
constexpr NonConstexpr2 nc2 = NonConstexpr2(); // ok, does not call constructor
constexpr NonConstexpr2a nc2a = NonConstexpr2a(); // ok, does not call constructor
constexpr int nc2_a = NonConstexpr2().nl.a; // ok
constexpr int nc2a_a = NonConstexpr2a().a; // ok
struct Helper {
  friend constexpr NonConstexpr1::NonConstexpr1(); // expected-error {{follows non-constexpr declaration}}
  friend constexpr NonConstexpr2::NonConstexpr2(); // expected-error {{follows non-constexpr declaration}}
};

struct Constexpr1 {};
constexpr Constexpr1 c1 = Constexpr1(); // ok
struct NonConstexpr3 : virtual Constexpr1 {}; // expected-note {{struct with virtual base}} expected-note {{declared here}}
constexpr NonConstexpr3 nc3 = NonConstexpr3(); // expected-error {{non-literal type 'const NonConstexpr3'}}

struct Constexpr2 {
  int a = 0;
};
constexpr Constexpr2 c2 = Constexpr2(); // ok

int n;
struct Member {
  Member() : a(n) {}
  constexpr Member(int&a) : a(a) {}
  int &a;
};
struct NonConstexpr4 { // expected-note {{here}}
  Member m;
};
constexpr NonConstexpr4 nc4 = NonConstexpr4(); // expected-error {{constant expression}} expected-note {{non-constexpr constructor 'NonConstexpr4'}}
struct Constexpr3 {
  constexpr Constexpr3() : m(n) {}
  Member m;
};
constexpr Constexpr3 c3 = Constexpr3(); // ok
struct Constexpr4 {
  Constexpr3 m;
};
constexpr Constexpr4 c4 = Constexpr4(); // ok


// This rule breaks some legal C++98 programs!
struct A {}; // expected-note {{here}}
struct B {
  friend A::A(); // expected-error {{non-constexpr declaration of 'A' follows constexpr declaration}}
};

namespace UnionCtors {
  union A { // expected-note {{here}}
    int a;
    int b;
  };
  union B {
    int a;
    int b = 5;
  };
  union C {
    int a = 5;
    int b;
  };
  struct D {
    union {
      int a = 5;
      int b;
    };
    union {
      int c;
      int d = 5;
    };
  };
  struct E { // expected-note {{here}}
    union {
      int a;
      int b;
    };
  };

  struct Test {
    friend constexpr A::A() noexcept; // expected-error {{follows non-constexpr declaration}}
    friend constexpr B::B() noexcept;
    friend constexpr C::C() noexcept;
    friend constexpr D::D() noexcept;
    friend constexpr E::E() noexcept; // expected-error {{follows non-constexpr declaration}}
  };
}

namespace PR48763 {
  // FIXME: We implement a speculative wording fix here: if a class inherits a
  // default constructor and doesn't declare one itself, we declare an default
  // constructor implicitly. This allows us to meanignfully reason about
  // whether that default constructor is constexpr, trivial, and so on.
  struct A { constexpr A() {} };
  struct B : A {
    using A::A;
    constexpr B(int) {}
  };
  struct C { B b; };
  constexpr C c;

  struct D { int n; };
  struct E : D { using D::D; E(int); };
  static_assert(E().n == 0, "");
  static_assert(E{}.n == 0, "");

  struct F { E e; };
  static_assert(F().e.n == 0, "");
  static_assert(F{}.e.n == 0, "");

  union U { E e; };
  U u; // OK, trivial default constructor

  struct G { G(); };
  struct H : D { using D::D; H(int); G g; };
  union V { H h; }; // expected-note {{field 'h' has a non-trivial default constructor}}
  V v; // expected-error {{deleted}}
}

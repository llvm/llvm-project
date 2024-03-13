// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

// If the implicitly-defined constructor would satisfy the requirements of a
// constexpr constructor, the implicitly-defined constructor is constexpr.
struct Constexpr1 {
  constexpr Constexpr1() : n(0) {}
  int n;
};
constexpr Constexpr1 c1a = Constexpr1(Constexpr1()); // ok
constexpr Constexpr1 c1b = Constexpr1(Constexpr1(c1a)); // ok

struct Constexpr2 {
  Constexpr1 ce1;
  constexpr Constexpr2() = default;
  constexpr Constexpr2(const Constexpr2 &o) : ce1(o.ce1) {}
  // no move constructor
};

constexpr Constexpr2 c2a = Constexpr2(Constexpr2()); // ok
constexpr Constexpr2 c2b = Constexpr2(Constexpr2(c2a)); // ok

struct Constexpr3 {
  Constexpr2 ce2;
  // all special constructors are constexpr, move ctor calls ce2's copy ctor
};

constexpr Constexpr3 c3a = Constexpr3(Constexpr3()); // ok
constexpr Constexpr3 c3b = Constexpr3(Constexpr3(c3a)); // ok

struct NonConstexprCopy {
  constexpr NonConstexprCopy() = default;
  NonConstexprCopy(const NonConstexprCopy &);
  constexpr NonConstexprCopy(NonConstexprCopy &&) = default;

  int n = 42;
};

NonConstexprCopy::NonConstexprCopy(const NonConstexprCopy &) = default; // expected-note {{here}}

constexpr NonConstexprCopy ncc1 = NonConstexprCopy(NonConstexprCopy()); // ok
constexpr NonConstexprCopy ncc2 = ncc1; // expected-error {{constant expression}} expected-note {{non-constexpr constructor}}

struct NonConstexprDefault {
  NonConstexprDefault() = default;
  constexpr NonConstexprDefault(int n) : n(n) {}
  int n;
};
struct Constexpr4 {
  NonConstexprDefault ncd;
};

constexpr NonConstexprDefault ncd = NonConstexprDefault(NonConstexprDefault(1));
constexpr Constexpr4 c4a = { ncd };
constexpr Constexpr4 c4b = Constexpr4(c4a);
constexpr Constexpr4 c4c = Constexpr4(static_cast<Constexpr4&&>(const_cast<Constexpr4&>(c4b)));

struct Constexpr5Base {};
struct Constexpr5 : Constexpr5Base { constexpr Constexpr5() {} };
constexpr Constexpr5 ce5move = Constexpr5();
constexpr Constexpr5 ce5copy = ce5move;

// An explicitly-defaulted constructor doesn't become constexpr until the end of
// its class. Make sure we note that the class has a constexpr constructor when
// that happens.
namespace PR13052 {
  template<typename T> struct S {
    S() = default; // expected-note 2{{here}}
    S(S&&) = default;
    S(const S&) = default;
    T t;
  };

  struct U {
    U() = default;
    U(U&&) = default;
    U(const U&) = default;
  };

  struct V {
    V(); // expected-note {{here}}
    V(V&&) = default;
    V(const V&) = default;
  };

  struct W {
    W(); // expected-note {{here}}
  };

  static_assert(__is_literal_type(U), "");
  static_assert(!__is_literal_type(V), "");
  static_assert(!__is_literal_type(W), "");
  static_assert(__is_literal_type(S<U>), "");
  static_assert(!__is_literal_type(S<V>), "");
  static_assert(!__is_literal_type(S<W>), "");

  struct X {
    friend constexpr U::U() noexcept;
    friend constexpr U::U(U&&) noexcept;
    friend constexpr U::U(const U&) noexcept;
    friend constexpr V::V(); // expected-error {{follows non-constexpr declaration}}
    friend constexpr V::V(V&&) noexcept;
    friend constexpr V::V(const V&) noexcept;
    friend constexpr W::W(); // expected-error {{follows non-constexpr declaration}}
    friend constexpr W::W(W&&) noexcept;
    friend constexpr W::W(const W&) noexcept;
    friend constexpr S<U>::S() noexcept;
    friend constexpr S<U>::S(S<U>&&) noexcept;
    friend constexpr S<U>::S(const S<U>&) noexcept;
    friend constexpr S<V>::S(); // expected-error {{follows non-constexpr declaration}}
    friend constexpr S<V>::S(S<V>&&) noexcept;
    friend constexpr S<V>::S(const S<V>&) noexcept;
    friend constexpr S<W>::S(); // expected-error {{follows non-constexpr declaration}}
    friend constexpr S<W>::S(S<W>&&) noexcept;
    friend constexpr S<W>::S(const S<W>&) noexcept;
  };
}

namespace Mutable {
  struct A {
    constexpr A(A &);
    A(const A &);
  };
  struct B {
    constexpr B(const B &) = default; // ok
    mutable A a;
  };
  struct C {
    constexpr C(const C &) = default; // expected-error {{not constexpr}}
    A a;
  };
}

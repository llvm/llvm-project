// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1z %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

namespace dr1715 { // dr1715: 3.9
#if __cplusplus >= 201103L
  struct B {
    template<class T> B(T, typename T::Q);
  };

  class S {
    using Q = int;
    template<class T> friend B::B(T, typename T::Q);
  };

  struct D : B {
    using B::B;
  };
  struct E : B { // expected-note 2{{candidate}}
    template<class T> E(T t, typename T::Q q) : B(t, q) {} // expected-note {{'Q' is a private member}}
  };

  B b(S(), 1);
  D d(S(), 2);
  E e(S(), 3); // expected-error {{no match}}
#endif
}

namespace dr1722 { // dr1722: 9
#if __cplusplus >= 201103L
void f() {
  const auto lambda = [](int x) { return x + 1; };
  // Without the DR applied, this static_assert would fail.
  static_assert(
      noexcept((int (*)(int))(lambda)),
      "Lambda-to-function-pointer conversion is expected to be noexcept");
}
#endif
} // namespace dr1722

namespace dr1734 { // dr1734: no
#if __cplusplus >= 201103L
struct A {
    A(const A&) = delete;
};
// FIXME: 'A' should not be trivially copyable because the class lacks at least
// one non-deleted copy constructor, move constructor, copy assignment
// operator, or move assignment operator.
static_assert(__is_trivially_copyable(A), "");
#endif
}

namespace dr1736 { // dr1736: 3.9
#if __cplusplus >= 201103L
struct S {
  template <class T> S(T t) {
    struct L : S {
      using S::S;
    };
    typename T::type value; // expected-error {{no member}}
    L l(value); // expected-note {{instantiation of}}
  }
};
struct Q { typedef int type; } q;
S s(q); // expected-note {{instantiation of}}
#endif
}

namespace dr1753 { // dr1753: 11
  typedef int T;
  struct A { typedef int T; };
  namespace B { typedef int T; }

  void f(T n) {
    n.~T();
    n.T::~T();

    n.dr1753::~T(); // expected-error {{'dr1753' does not refer to a type name in pseudo-destructor}}
    n.dr1753::T::~T();

    n.A::~T(); // expected-error {{the type of object expression ('T' (aka 'int')) does not match the type being destroyed ('A') in pseudo-destructor expression}}
    n.A::T::~T();

    n.B::~T(); // expected-error {{'B' does not refer to a type name in pseudo-destructor expression}}
    n.B::T::~T();

  #if __cplusplus >= 201103L
    n.decltype(n)::~T(); // expected-error {{not a class, namespace, or enumeration}}
    n.T::~decltype(n)(); // expected-error {{expected a class name after '~'}}
    n.~decltype(n)(); // OK
  #endif
  }
}

namespace dr1756 { // dr1756: 3.7
#if __cplusplus >= 201103L
  // Direct-list-initialization of a non-class object
  
  int a{0};
  
  struct X { operator int(); } x;
  int b{x};
#endif
}

namespace dr1758 { // dr1758: 3.7
#if __cplusplus >= 201103L
  // Explicit conversion in copy/move list initialization

  struct X { X(); };
  struct Y { explicit operator X(); } y;
  X x{y};

  struct A {
    A() {}
    A(const A &) {}
  };
  struct B {
    operator A() { return A(); }
  } b;
  A a{b};
#endif
}

namespace dr1762 { // dr1762: 14
#if __cplusplus >= 201103L
  float operator ""_E(const char *);
  // expected-error@+2 {{invalid suffix on literal; C++11 requires a space between literal and identifier}}
  // expected-warning@+1 {{user-defined literal suffixes not starting with '_' are reserved; no literal will invoke this operator}}
  float operator ""E(const char *);
#endif
}

namespace dr1778 { // dr1778: 9
  // Superseded by P1286R2.
#if __cplusplus >= 201103L
  struct A { A() noexcept(true) = default; };
  struct B { B() noexcept(false) = default; };
  static_assert(noexcept(A()), "");
  static_assert(!noexcept(B()), "");

  struct C { A a; C() noexcept(false) = default; };
  struct D { B b; D() noexcept(true) = default; };
  static_assert(!noexcept(C()), "");
  static_assert(noexcept(D()), "");
#endif
}

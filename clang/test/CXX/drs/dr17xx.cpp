// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1z %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

#if __cplusplus < 201103L
// expected-no-diagnostics
#endif

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

// RUN: %clang_cc1 -std=c++11 -verify %s -pedantic

namespace PR31692 {
  struct A {
    struct X { int n = 0; } x;
    // Trigger construction of X() from a SFINAE context. This must not mark
    // any part of X as invalid.
    static_assert(!__is_constructible(X), "");
    // Check that X::n is not marked invalid.
    double &r = x.n; // expected-error {{non-const lvalue reference to type 'double' cannot bind to a value of unrelated type 'int'}}
  };
  // A::X can now be default-constructed.
  static_assert(__is_constructible(A::X), "");
}

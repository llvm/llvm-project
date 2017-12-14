// RUN: %clang_cc1 -std=c++2a -verify %s -fcxx-exceptions -triple=x86_64-linux-gnu

namespace ThreeWayComparison {
  struct A {
    int n;
    constexpr friend int operator<=>(const A &a, const A &b) {
      return a.n < b.n ? -1 : a.n > b.n ? 1 : 0;
    }
  };
  static_assert(A{1} <=> A{2} < 0);
  static_assert(A{2} <=> A{1} > 0);
  static_assert(A{2} <=> A{2} == 0);

  // Note: not yet supported.
  static_assert(1 <=> 2 < 0); // expected-error {{invalid operands}}
  static_assert(2 <=> 1 > 0); // expected-error {{invalid operands}}
  static_assert(1 <=> 1 == 0); // expected-error {{invalid operands}}
  constexpr int k = (1 <=> 1, 0);
  // expected-error@-1 {{constexpr variable 'k' must be initialized by a constant expression}}
  // expected-warning@-2 {{three-way comparison result unused}}

  constexpr void f() { // expected-error {{constant expression}}
    void(1 <=> 1); // expected-note {{constant expression}}
  }

  // TODO: defaulted operator <=>
}

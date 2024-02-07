// RUN: %clang_cc1 -std=c++23 -x c++ %s -verify

struct A{};
struct B{ explicit operator bool() { return true; } };

void f(int x, int y) {
  [[assume(true)]];
  [[assume(1)]];
  [[assume(1.0)]];
  [[assume(1 + 2 == 3)]];
  [[assume((x = 3))]];
  [[assume(x ? 1 : 2)]];
  [[assume(x && y)]];
  [[assume(x++)]];
  [[assume(++x)]];
  [[assume([]{ return true; }())]];
  [[assume(B{})]];
  [[assume(true)]] [[assume(true)]];

  [[assume((1, 2))]]; // expected-warning {{has no effect}}

  [[assume]]; // expected-error {{takes one argument}}
  [[assume(z)]]; // expected-error {{undeclared identifier}}
  [[assume(x = 2)]]; // expected-error {{requires parentheses}}
  [[assume(2, 3)]]; // expected-error {{requires parentheses}}
  [[assume(A{})]]; // expected-error {{not contextually convertible to 'bool'}}
  [[assume(true)]] if (true) {} // expected-error {{only applies to empty statements}}
  [[assume(true)]] {} // expected-error {{only applies to empty statements}}
  [[assume(true)]] for (;false;) {} // expected-error {{only applies to empty statements}}
  [[assume(true)]] while (false) {} // expected-error {{only applies to empty statements}}
  [[assume(true)]] label:; // expected-error {{cannot be applied to a declaration}}
  [[assume(true)]] goto label; // expected-error {{only applies to empty statements}}
}
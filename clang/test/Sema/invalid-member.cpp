// RUN: %clang_cc1 -verify -fsyntax-only %s
void foo(); // expected-note {{requires 0 arguments}}
class X {
  decltype(foo(42)) invalid; // expected-error {{no matching function}}
};
// Should be able to evaluate sizeof without crashing.
static_assert(sizeof(X) == 1, "No valid members");

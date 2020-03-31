// RUN: %clang_cc1 -verify -fsyntax-only -fno-recovery-ast %s
// RUN: %clang_cc1 -verify -fsyntax-only -frecovery-ast %s

void foo(); // expected-note 2{{requires 0 arguments}}
class X {
  decltype(foo(42)) invalid; // expected-error {{no matching function}}
};
// Should be able to evaluate sizeof without crashing.
static_assert(sizeof(X) == 1, "No valid members");

class Y {
  typeof(foo(42)) invalid; // expected-error {{no matching function}}
};
// Should be able to evaluate sizeof without crashing.
static_assert(sizeof(Y) == 1, "No valid members");

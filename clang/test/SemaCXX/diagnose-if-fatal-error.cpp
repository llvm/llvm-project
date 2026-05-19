// RUN: %clang_cc1 -fsyntax-only -std=c++17 -ferror-limit 1 -verify %s
// Regression test for #197625: diagnose_if condition becomes value-dependent
// via RecoveryExpr after fatal error, hitting an assertion in EvaluateWithSubstitution.

template <typename T> struct S {
  T val;
  constexpr T get() const { return val; }
};

void bad(int i) __attribute__((diagnose_if(i, "bad", "error"))); // expected-note {{from 'diagnose_if'}}
void blast() {
  bad(1); // expected-error {{bad}}
  bad(1); // expected-error@* {{too many errors}}
}

struct Foo {
  Foo(S<int> s) __attribute__((diagnose_if(s.get() == 1, "x", "warning"))) {}
};
void run() { Foo{S<int>{1}}; }

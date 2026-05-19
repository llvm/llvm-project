// RUN: %clang_cc1 -fsyntax-only -std=c++17 -ferror-limit 1 -verify %s
// Regression test for #197625: diagnose_if condition becomes value-dependent
// via RecoveryExpr after fatal error, hitting an assertion in EvaluateWithSubstitution.

template <typename T> struct S {
  T val;
  constexpr T get() const { return val; }
};

// The isValueDependent() guard must not suppress diagnose_if after instantiation.
struct Bar {
  template <typename T>
  Bar(S<T> s) __attribute__((diagnose_if(s.get() == 1, "one", "error"))) {} // expected-note {{from 'diagnose_if'}}
};
void instantiated() { Bar{S<int>{1}}; } // expected-error {{one}}

// After fatal error, the guard must skip evaluation instead of crashing.
void bad(int i) __attribute__((diagnose_if(i, "bad", "error")));
void blast() {
  bad(1); // expected-error@* {{too many errors}}
}

struct Foo {
  Foo(S<int> s) __attribute__((diagnose_if(s.get() == 1, "x", "warning"))) {}
};
void run() { Foo{S<int>{1}}; }

// RUN: %clang_cc1 -fsyntax-only -verify %s

// Ensure that we don't crash if errors are suppressed by an error limit.
// RUN: not %clang_cc1 -fsyntax-only -ferror-limit 1 %s

template <typename T> struct S {
  T val;
  constexpr T get() const { return val; }
};

struct Bar {
  template <typename T>
  Bar(S<T> s) __attribute__((diagnose_if(s.get() == 1, "one", "error"))) {} // expected-note {{from 'diagnose_if'}}
};
void instantiated() { Bar{S<long>{1}}; } // expected-error {{one}}

void bad(int i) __attribute__((diagnose_if(i, "bad", "error"))); // expected-note {{from 'diagnose_if'}}
void blast() { bad(1); } // expected-error {{bad}}

struct Foo {
  Foo(S<int> s) __attribute__((diagnose_if(s.get() == 1, "x", "warning"))) {} // expected-note {{from 'diagnose_if'}}
};
void run() { Foo{S<int>{1}}; } // expected-warning {{x}}

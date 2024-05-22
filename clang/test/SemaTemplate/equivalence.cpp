// RUN: %clang_cc1 -fsyntax-only -verify %s

// Ensure we undo the rewrite from `a++` to a binary `a ++ 0` before profiling.
namespace PostIncDec {
  // Increment / decrement as UnaryOperator.
  template<typename T> auto inc(T &a) -> decltype(a++) {} // expected-note {{previous}}
  template<typename T> auto dec(T &a) -> decltype(a--) {} // expected-note {{previous}}

  struct X {};
  void operator++(X&, int);
  void operator--(X&, int);
  // Increment / decrement as CXXOverloadedCallExpr. These are redefinitions.
  template<typename T> auto inc(T &a) -> decltype(a++) {} // expected-error {{redefinition}} expected-note {{candidate}}
  template<typename T> auto dec(T &a) -> decltype(a--) {} // expected-error {{redefinition}} expected-note {{candidate}}

  // These are not ambiguous calls.
  void f(X x) {
    // FIXME: Don't produce these follow-on errors.
    inc(x); // expected-error {{no match}}
    dec(x); // expected-error {{no match}}
  }
}

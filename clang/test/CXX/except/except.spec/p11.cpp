// RUN: %clang_cc1 -std=c++11 -fexceptions -fcxx-exceptions -fsyntax-only -verify %s

// This is the "let the user shoot themselves in the foot" clause.
void f() noexcept { // expected-note {{function declared non-throwing here}}
  throw 0; // expected-warning {{has a non-throwing exception specification but}} 
}
void g() throw() { // expected-note {{function declared non-throwing here}}
  throw 0; // expected-warning {{has a non-throwing exception specification but}} 
}
void h() throw(int) {
  throw 0.0; // no-error
}

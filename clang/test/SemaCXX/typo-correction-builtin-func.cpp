// RUN: %clang_cc1 -fsyntax-only -verify %s

// Test that clang does not emit 'declared here' note for builtin functions that don't have a declaration in source.

void t0() {
  constexpr float A = __builtin_isinfinity(); // expected-error {{use of undeclared identifier '__builtin_isinfinity'; did you mean '__builtin_isfinite'?}}
                                              // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
}

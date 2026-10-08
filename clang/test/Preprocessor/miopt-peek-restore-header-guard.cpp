// RUN: %clang_cc1 -std=c++20 -fsyntax-only -Wheader-guard -verify %s

// Regression test for Lexer::peekNextPPToken() saving/restoring MIOpt state.
// In C++20 module mode, the preprocessor peeks the first pp-token in the main
// file before lexing begins. That must not perturb MIOpt state used for
// header-guard analysis.

#ifndef GOOD_GUARD
#define BAD_GUARD
#endif
// expected-warning@-3 {{'GOOD_GUARD' is used as a header guard here, followed by #define of a different macro}}
// expected-note@-3 {{'BAD_GUARD' is defined here; did you mean 'GOOD_GUARD'?}}

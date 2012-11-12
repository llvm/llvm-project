// RUN: %clang_cc1 -fsyntax-only -verify %s 
// XFAIL: *

// A template-parameter shall not be used in its own default argument.
template<typename T = typename T::type> struct X; // expected-error{{default}}

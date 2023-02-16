// RUN: %clang_cc1 -std=c++20 -verify %s

// A named module shall contain exactly one module interface unit.
module M; // expected-error {{module 'M' not found}}

// FIXME: How do we ensure there is not more than one?

// RUN: %clang_cc1 -fmodules-ts -verify %s

// A named module shall contain exactly one module interface unit.
module M; // expected-error {{module 'M' not found}}

// FIXME: How do we ensure there is not more than one?

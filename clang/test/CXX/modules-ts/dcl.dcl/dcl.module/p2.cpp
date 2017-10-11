// RUN: %clang_cc1 -fmodules-ts -verify %s

// A named module shall contain exactly one module interface unit.
module M; // expected-error {{definition of module 'M' is not available; use -fmodule-file= to specify path to precompiled module interface}}

// FIXME: How do we ensure there is not more than one?

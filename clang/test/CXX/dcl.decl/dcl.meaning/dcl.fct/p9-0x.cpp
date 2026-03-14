// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// FIXME: We should catch the case of tag with an incomplete type here (which
// will necessarily be ill-formed as a trailing return type for a function
// definition), and recover with a "type cannot be defined in a trailing return
// type" error.
auto j() -> enum { e3 }; // expected-error{{unnamed enumeration must be a definition}}

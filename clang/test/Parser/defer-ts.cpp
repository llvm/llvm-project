// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fdefer-ts -verify %s

// FIXME: Do we want to support 'defer' in C++? For now, we just reject
// it in the parser if '-fdefer-ts' is passed, but if we decide *not* to
// support it in C++, then we should probably strip out and warn about
// that flag in the driver (or frontend?) instead.
void f() {
  defer {} // expected-error {{'defer' statements are only supported in C}}
}

// RUN: %clang_cc1 -fsyntax-only -verify %s

bool f() {
  return __builtin_types_compatible_p(int, const int); // expected-error{{C++}}
}

// RUN: %clang_cc1 -fsyntax-only -verify -x c++ %s
// RUN: %clang_cc1 -fsyntax-only -x c %s

// Test that GNU C extension __builtin_types_compatible_p() is not available in C++ mode.

int f(void) {
  return __builtin_types_compatible_p(int, const int); // expected-error{{expected '(' for function-style cast or type construction}} \
                                                       // expected-error{{expected expression}}
}

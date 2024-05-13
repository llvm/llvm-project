// RUN: %clang_cc1 -fsyntax-only -verify %s

inline void __attribute__((artificial)) foo(void) {}
void __attribute__((artificial)) bar(void) {} // expected-warning {{'artificial' attribute only applies to inline functions}}

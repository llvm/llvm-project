// RUN: %clang_cc1 -fsyntax-only -verify %s

inline void __attribute__((artificial)) foo() {}
void __attribute__((artificial)) bar() {} // expected-warning {{'artificial' attribute only applies to inline functions}}

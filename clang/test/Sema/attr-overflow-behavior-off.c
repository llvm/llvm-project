// RUN: %clang_cc1 %s -Winteger-overflow -Wno-unused-value -verify -fsyntax-only

typedef int __attribute__((overflow_behavior(wrap))) wrap_int; // expected-warning {{'overflow_behavior' attribute is ignored because it is not enabled;}}

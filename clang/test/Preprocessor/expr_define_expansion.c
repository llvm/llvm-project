// RUN: %clang_cc1 %s -E -CC -pedantic -verify
// expected-no-diagnostics

#define FOO && 1
#if defined FOO FOO
#endif

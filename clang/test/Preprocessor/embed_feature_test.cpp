// RUN: %clang_cc1 %s -E -CC -verify
// RUN: %clang_cc1 -x c %s -E -CC -verify
// expected-no-diagnostics

#if !defined(__has_embed)
#error 1
#endif

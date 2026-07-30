// RUN: %clang_cc1 -triple i386-linux-android -fsyntax-only -verify %s
// expected-no-diagnostics

extern int a1_0[sizeof(long double) == 8 ? 1 : -1];
extern int a1_i[__alignof(long double) == 4 ? 1 : -1];


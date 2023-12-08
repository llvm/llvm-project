// RUN: %clang_cc1 -fsyntax-only -verify -std=c2x %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c23 %s
// expected-no-diagnostics

_Static_assert(__STDC_VERSION__ == 202311L, "Incorrect __STDC_VERSION__");

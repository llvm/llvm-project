// RUN: %clang_cc1 -fsyntax-only -verify -std=c17 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c18 %s
// expected-no-diagnostics

_Static_assert(__STDC_VERSION__ == 201710L, "Incorrect __STDC_VERSION__");

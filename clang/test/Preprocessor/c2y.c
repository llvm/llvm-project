// RUN: %clang_cc1 -fsyntax-only -verify -std=c2y %s
// expected-no-diagnostics

// FIXME: Set this to test the correct value once that value is set by WG14.
static_assert(__STDC_VERSION__ > 202311L, "Incorrect __STDC_VERSION__");


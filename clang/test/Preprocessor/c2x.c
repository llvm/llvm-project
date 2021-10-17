// RUN: %clang_cc1 -fsyntax-only -verify -std=c2x %s
// expected-no-diagnostics

// FIXME: Test the correct value once C23 ships.
_Static_assert(__STDC_VERSION__ > 201710L, "Incorrect __STDC_VERSION__");

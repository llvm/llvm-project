// RUN: %clang_cc1 -std=c2x -verify %s
// expected-no-diagnostics

/* WG14 N2670: yes
 * Zeros compare equal
 */
_Static_assert(-1 * 0.0 == 0.0, "");
_Static_assert(!(-1 * 0.0 < 0.0), "");

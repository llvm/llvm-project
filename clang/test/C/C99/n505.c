// RUN: %clang_cc1 -verify %s

/* WG14 N505: Yes
 * Make qualifiers idempotent
 */
const const int i = 12; // expected-warning {{duplicate 'const' declaration specifier}}
typedef const int cint;
const cint j = 12;


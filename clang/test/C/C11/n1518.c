// RUN: %clang_cc1 -verify=c11,both -std=c11 %s
// RUN: %clang_cc1 -verify=c23,both -std=c23 %s

/* WG14 N1518: Clang 15
 * Recommendations for extended identifier characters for C and C++
 *
 * This paper effectively adopts UAX #31, which was later officially adopted
 * for C23 via WG14 N2836 and supersedes N1518.
 */

// This file takes test cases from clang/test/C/C23/n2836_n2939.c.
// This file contains Unicode characters; please do not "fix" them!

// This was fine in C11, is now an error in C23.
extern int ٢;  // c23-error {{character <U+0662> not allowed at the start of an identifier}} \
                  c23-warning {{declaration does not declare anything}}

// This was an error in C11 but is an extension in C23.
extern int ∞;  // c11-error {{unexpected character <U+221E>}} \
                  c11-warning {{declaration does not declare anything}} \
                  c23-warning {{mathematical notation character <U+221E> in an identifier is a Clang extension}}

int \u1DC0;  // both-error {{expected identifier or '('}}
int e\u1DC0; // Ok

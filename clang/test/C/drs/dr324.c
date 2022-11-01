/* RUN: %clang_cc1 -std=c89 -fsyntax-only -fms-extensions -pedantic -verify %s
   RUN: %clang_cc1 -std=c99 -fsyntax-only -fms-extensions -pedantic -verify %s
   RUN: %clang_cc1 -std=c11 -fsyntax-only -fms-extensions -pedantic -verify %s
   RUN: %clang_cc1 -std=c17 -fsyntax-only -fms-extensions -pedantic -verify %s
   RUN: %clang_cc1 -std=c2x -fsyntax-only -fms-extensions -pedantic -verify %s
 */

/* WG14 DR324: yes
 * Tokenization obscurities
 */

/* We need to diagnose an unknown escape sequence in a string or character
 * literal, but not within a header-name terminal.
 */
const char *lit_str = "\y"; /* expected-warning {{unknown escape sequence '\y'}} */
char lit_char = '\y';       /* expected-warning {{unknown escape sequence '\y'}} */

/* This gets trickier in a pragma where there are implementation-defined
 * locations that may use a header-name production. The first pragma below
 * is using \d but it's in a header-name use rather than a string-literal use.
 * The second pragma is a string-literal and so the \d is invalid there.
 */
#ifdef _WIN32
/* This test only makes sense on Windows targets where the backslash is a valid
 * path separator.
 */
#pragma GCC dependency "oops\..\dr0xx.c"
#endif
#pragma message("this has a \t tab escape and an invalid \d escape") /* expected-warning {{this has a 	 tab escape and an invalid d escape}}
                                                                        expected-warning {{unknown escape sequence '\d'}}
                                                                      */

/*
 * Note, this tests the behavior of a non-empty source file that ends with a
 * partial preprocessing token such as an unterminated string or character
 * literal. Thus, it is important that no code be added after this test case.
 */
/* expected-error@+3 {{expected identifier or '('}}
   expected-warning@+3 {{missing terminating ' character}}
 */
't

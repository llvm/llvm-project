/* RUN: %clang_cc1 -fsyntax-only -verify %s
*/

void foo(int A) { /* expected-note {{previous definition is here}} */
  int A; /* expected-error {{redefinition of 'A'}} */
}

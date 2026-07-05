/* RUN: %clang_cc1 -std=c89 -fsyntax-only -Wuninitialized -verify %s
   RUN: %clang_cc1 -std=c99 -fsyntax-only -Wuninitialized -verify %s
   RUN: %clang_cc1 -std=c11 -fsyntax-only -Wuninitialized -verify %s
   RUN: %clang_cc1 -std=c17 -fsyntax-only -Wuninitialized -verify %s
   RUN: %clang_cc1 -std=c2x -fsyntax-only -Wuninitialized -verify %s
 */

/* WG14 DR338: yes
 * C99 seems to exclude indeterminate value from being an uninitialized register
 *
 * Note, because we're relying on -Wuninitialized, this test has to live in its
 * own file. That analysis will not run if the file has other errors in it.
 */
int dr338(void) {
  unsigned char uc;   /* expected-note {{initialize the variable 'uc' to silence this warning}} */
  return uc + 1 >= 0; /* expected-warning {{variable 'uc' is uninitialized when used here}} */
}

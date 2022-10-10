// RUN: %clang_cc1 -verify %s
// RUN: %clang_cc1 -verify=c2x -std=c2x %s

/* WG14 N636: yes
 * remove implicit function declaration
 */

void test(void) {
  frobble(); // expected-error {{call to undeclared function 'frobble'; ISO C99 and later do not support implicit function declarations}} \
                c2x-error {{undeclared identifier 'frobble'}}
}


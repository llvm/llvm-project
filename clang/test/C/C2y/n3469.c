// RUN: %clang_cc1 -fsyntax-only -std=c2y -verify %s

/* WG14 N3469: Clang 21
 * The Big Array Size Survey
 *
 * This renames _Lengthof to _Countof.
 */

void test() {
  (void)_Countof(int[12]); // Ok
  (void)_Lengthof(int[12]); // expected-error {{use of undeclared identifier '_Lengthof'}} \
                               expected-error {{expected expression}}
}


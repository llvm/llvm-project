// RUN: %clang_cc1 -verify -std=c2y -Wall -pedantic -Wno-unused -Wno-deprecated %s

/* WG14 N3715: No
 * static_assert expressions
 *
 * A successful static_assert is also an expression of type void.
 *
 * FIXME: Clang doesn't yet implement this paper.
 */

// FIXME: Should accept these.
void test() {
  _Generic(static_assert(true, ""), void: (void)0);  // expected-error {{expected expression}}
  _Generic(static_assert(true), void: (void)0);      // expected-error {{expected expression}}
  _Generic(_Static_assert(true, ""), void: (void)0); // expected-error {{expected expression}}
  _Generic(_Static_assert(true), void: (void)0);     // expected-error {{expected expression}}
}

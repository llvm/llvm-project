// RUN: %clang_cc1 -verify -std=c2y -Wall -pedantic -Wvla -Wno-unused %s

/* WG14 N3787: No
 * Array constant expressions
 *
 * FIXME: Clang doesn't yet implement this paper.
 */

constexpr int p[3] = { 2, 3, 5 };

void test() {
  // FIXME: This should not be a VLA, p[0] is a valid integer constant
  // expression.
  float y[p[0]]; // expected-warning {{variable length array used}}

  // Is a VLA, p[3] is not a valid constant expression because of the out-of-
  // bounds read.
  float z[p[3]]; // expected-warning {{variable length array used}}
}


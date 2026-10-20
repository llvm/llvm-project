// RUN: %clang_cc1 -verify -std=c2y -Wall -pedantic %s

/* WG14 N3348: No
 * Matching of Multi-Dimensional Arrays in Generic Selection Expressions
 *
 * This allows use of * in a _Generic association as a placeholder for any size
 * value.
 *
 * FIXME: Clang doesn't yet implement this paper. When we do implement it, we
 * should expose the functionality in earlier language modes (C89) for
 * compatibility with GCC.
 */

void test(int n, int m) {
  static_assert(1 == _Generic(int[3][2], int[3][*]: 1, int[2][*]: 0));  /* expected-error {{star modifier used outside of function prototype}}
                                                                           expected-error {{array has incomplete element type 'int[]'}}
                                                                         */
  static_assert(1 == _Generic(int[3][2], int[*][2]: 1, int[*][3]: 0));  // expected-error {{star modifier used outside of function prototype}}
  static_assert(1 == _Generic(int[3][n], int[3][*]: 1, int[2][*]: 0));  /* expected-error {{star modifier used outside of function prototype}}
                                                                           expected-error {{array has incomplete element type 'int[]'}}
                                                                         */
  static_assert(1 == _Generic(int[n][m], int[*][*]: 1, char[*][*]: 0)); /* expected-error 2 {{star modifier used outside of function prototype}}
                                                                           expected-error {{array has incomplete element type 'int[]'}}
                                                                         */
  static_assert(1 == _Generic(int(*)[2], int(*)[*]: 1));                // expected-error {{star modifier used outside of function prototype}}
}

void questionable() {
  // GCC accepts this despite the * appearing outside of a generic association,
  // but it's not clear whether that's intentionally supported or an oversight.
  // It gives a warning about * being used outside of a declaration, but not
  // with an associated warning group.
  static_assert(1 == _Generic(int[*][*], int[2][100]: 1)); /* expected-error 2 {{star modifier used outside of function prototype}}
                                                              expected-error {{array has incomplete element type 'int[]'}}
                                                            */
  // GCC claims this matches multiple associations, so the functionality seems
  // like it may be intended to work?
  (void)_Generic(int[*][*], /* expected-error 2 {{star modifier used outside of function prototype}}
                               expected-error {{array has incomplete element type 'int[]'}}
                             */
    int[2][100]: 1,
    int[3][1000]: 2,
  );
}

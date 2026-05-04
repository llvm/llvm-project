// RUN: %clang_cc1 -verify -std=c2y -Wall -pedantic -Wpre-c2y-compat %s
// RUN: %clang_cc1 -verify=pre-c2y -std=c23 -Wall -pedantic %s

/* WG14 N3260: Clang 17
 * Generic selection expression with a type operand
 */

static_assert(
  _Generic(
    const int, /* pre-c2y-warning {{passing a type argument as the first operand to '_Generic' is a C2y extension}}
                  expected-warning {{passing a type argument as the first operand to '_Generic' is incompatible with C standards before C2y}}
                */
    int : 0,
    const int : 1
  )
);

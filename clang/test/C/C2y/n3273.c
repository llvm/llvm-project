// RUN: %clang_cc1 -verify -std=c2y -Wall -pedantic -Wpre-c2y-compat %s
// RUN: %clang_cc1 -verify=pre-c2y -std=c23 -Wall -pedantic %s

/* WG14 N3273: Clang 3.5
 * alignof of an incomplete array type
 */

static_assert(
  alignof(int[]) == /* pre-c2y-warning {{'alignof' on an incomplete array type is a C2y extension}}
                       expected-warning {{'alignof' on an incomplete array type is incompatible with C standards before C2y}}
                     */
  alignof(int)
);


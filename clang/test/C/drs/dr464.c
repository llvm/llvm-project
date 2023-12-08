/* RUN: %clang_cc1 -std=c89 -verify -pedantic -Wno-c11-extensions %s
   RUN: %clang_cc1 -std=c99 -verify -pedantic -Wno-c11-extensions %s
   RUN: %clang_cc1 -std=c11 -verify -pedantic %s
   RUN: %clang_cc1 -std=c17 -verify -pedantic %s
   RUN: %clang_cc1 -std=c2x -verify -pedantic %s
 */

/* expected-no-diagnostics */

/* WG14 DR464: yes
 * Clarifying the Behavior of the #line Directive
 *
 * Note: the behavior described by this DR allows for two different
 * interpretations, but WG14 N2322 (adopted for C2x) adds a recommended
 * practice which is what we're testing our interpretation against.
 */
#line 10000
_Static_assert(__LI\
NE__ == 10000, "");

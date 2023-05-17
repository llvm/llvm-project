/* RUN: %clang_cc1 -std=c89 -verify -pedantic -Wno-c11-extensions %s
   RUN: %clang_cc1 -std=c99 -verify -pedantic -Wno-c11-extensions %s
   RUN: %clang_cc1 -std=c11 -verify -pedantic %s
   RUN: %clang_cc1 -std=c17 -verify -pedantic %s
   RUN: %clang_cc1 -std=c2x -verify -pedantic %s
 */

/* expected-no-diagnostics */

/* WG14 DR483: yes
 * __LINE__ and __FILE__ in macro replacement list
 *
 * The crux of this DR is to ensure that __LINE__ (and __FILE__) use in a macro
 * replacement list report the line and file of the expansion of that macro,
 * not the line and file of the macro definition itself.
 */
#line 500
#define MAC() __LINE__

#line 1000
_Static_assert(MAC() == 1000, "");


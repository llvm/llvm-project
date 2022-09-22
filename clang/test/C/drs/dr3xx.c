/* RUN: %clang_cc1 -std=c89 -fsyntax-only -Wvla -verify=expected,c89only -pedantic -Wno-c11-extensions %s
   RUN: %clang_cc1 -std=c99 -fsyntax-only -Wvla -verify -pedantic -Wno-c11-extensions %s
   RUN: %clang_cc1 -std=c11 -fsyntax-only -Wvla -verify -pedantic %s
   RUN: %clang_cc1 -std=c17 -fsyntax-only -Wvla -verify -pedantic %s
   RUN: %clang_cc1 -std=c2x -fsyntax-only -Wvla -verify -pedantic %s
 */

/* The following are DRs which do not require tests to demonstrate
 * conformance or nonconformance.
 *
 * WG14 DR300: yes
 * Translation-time expresssion evaluation
 *
 * WG14 DR301: yes
 * Meaning of FE_* macros in <fenv.h>
 *
 * WG14 DR303: yes
 * Breaking up the very long sentence describing preprocessing directive
 *
 * WG14 DR307: yes
 * Clarifiying arguments vs. parameters
 *
 * WG14 DR308: yes
 * Clarify that source files et al. need not be "files"
 *
 * WG14 DR310: yes
 * Add non-corner case example of trigraphs
 *
 * WG14 DR312: yes
 * Meaning of "known constant size"
 */


/* WG14 DR302: yes
 * Adding underscore to portable include file name character set
 */
#include "./abc_123.h"
#ifndef WE_SUPPORT_DR302
#error "Oh no, we don't support DR302 after all!"
#endif

/* WG14 DR304: yes
 * Clarifying illegal tokens in #if directives
 */
/* expected-error@+3 {{invalid token at start of a preprocessor expression}}
   expected-warning@+3 {{missing terminating ' character}}
 */
#if 'test
#endif

/* WG14 DR305: yes
 * Clarifying handling of keywords in #if directives
 */
#if int
#error "We definitely should not have gotten here"
#endif

/* WG14 DR306: yes
 * Clarifying that rescanning applies to object-like macros
 */
#define REPLACE 1
#define THIS REPLACE
#if THIS != 1
#error "We definitely should not have gotten here"
#endif

/* WG14 DR309: yes
 * Clarifying trigraph substitution
 */
int dr309??(1??) = { 1 }; /* expected-warning {{trigraph converted to '[' character}}
                             expected-warning {{trigraph converted to ']' character}}
                           */

/* WG14 DR311: yes
 * Definition of variably modified types
 */
void dr311(int x) {
  typedef int vla[x]; /* expected-warning {{variable length array}} */

  /* Ensure that a constant array of variable-length arrays are still
   * considered a variable-length array.
   */
  vla y[3]; /* expected-warning {{variable length array}} */
}

/* WG14 DR313: yes
 * Incomplete arrays of VLAs
 */
void dr313(int i) {
  int c[][i] = { 0 }; /* expected-error {{variable-sized object may not be initialized}}
                         expected-warning {{variable length array}}
                       */
}

/* WG14 DR315: yes
 * Implementation-defined bit-field types
 */
struct dr315_t {
  unsigned long long a : 37; /* c89only-warning {{'long long' is an extension when C99 mode is not enabled}} */
  unsigned long long b : 37; /* c89only-warning {{'long long' is an extension when C99 mode is not enabled}} */

  short c : 8;
  short d : 8;
} dr315;
_Static_assert(sizeof(dr315.a + dr315.b) == sizeof(unsigned long long), ""); /* c89only-warning {{'long long' is an extension when C99 mode is not enabled}} */
/* Demonstrate that integer promotions still happen when less than the width of
 * an int.
 */
_Static_assert(sizeof(dr315.c + dr315.d) == sizeof(int), "");

/* WG14 DR316: yes
 * Unprototyped function types
 */
#if __STDC_VERSION__ < 202000L
void dr316_1(a) int a; {}  /* expected-warning {{a function definition without a prototype is deprecated in all versions of C and is not supported in C2x}} */
void (*dr316_1_ptr)(int, int, int) = dr316_1;
#endif /* __STDC_VERSION__ < 202000L */

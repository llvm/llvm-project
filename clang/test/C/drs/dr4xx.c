/* RUN: %clang_cc1 -std=c89 -verify=expected,c89only -pedantic -Wno-c11-extensions %s
   RUN: %clang_cc1 -std=c99 -verify=expected,c99untilc2x -pedantic -Wno-c11-extensions %s
   RUN: %clang_cc1 -std=c11 -verify=expected,c99untilc2x -pedantic %s
   RUN: %clang_cc1 -std=c17 -verify=expected,c99untilc2x -pedantic %s
   RUN: %clang_cc1 -std=c2x -verify=expected,c2xandup -pedantic %s
 */

/* The following are DRs which do not require tests to demonstrate
 * conformance or nonconformance.
 */


/* WG14 DR423: partial
 * Defect Report relative to n1570: underspecification for qualified rvalues
 */

/* FIXME: this should pass because the qualifier on the return type should be
 * dropped when forming the function type.
 */
const int dr423_const(void);
int dr423_nonconst(void);
_Static_assert(__builtin_types_compatible_p(__typeof__(dr423_const), __typeof__(dr423_nonconst)), "fail"); /* expected-error {{fail}} */

void dr423_func(void) {
  const int i = 12;
  __typeof__(i) v1 = 12; /* expected-note {{variable 'v1' declared const here}} */
  __typeof__((const int)12) v2 = 12;

  v1 = 100; /* expected-error {{cannot assign to variable 'v1' with const-qualified type 'typeof (i)' (aka 'const int')}} */
  v2 = 100; /* Not an error; the qualifier was stripped. */
}


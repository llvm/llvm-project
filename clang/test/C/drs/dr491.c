/* RUN: %clang_cc1 -std=c89 -verify -Wreserved-macro-identifier %s
   RUN: %clang_cc1 -std=c99 -verify -Wreserved-macro-identifier %s
   RUN: %clang_cc1 -std=c11 -verify -Wreserved-macro-identifier %s
   RUN: %clang_cc1 -std=c17 -verify -Wreserved-macro-identifier %s
   RUN: %clang_cc1 -std=c2x -verify -Wreserved-macro-identifier %s
 */

/* WG14 DR491: partial
 * Concern with Keywords that Match Reserved Identifiers
 *
 * Claiming this as partial because we do not reject code using a reserved
 * identifier, but our reserved identifier code incorrectly identifies some
 * keywords as reserved identifiers for macro names, but not others.
 */

#define const const
#define int int
#define restrict restrict

/* FIXME: none of these should diagnose the macro name as a reserved
 * identifier per C2x 6.4.2p7 (similar wording existed in earlier standard
 * versions).
 */
#define _Static_assert _Static_assert  /* expected-warning {{macro name is a reserved identifier}} */
#define _Alignof(x) _Alignof(x)        /* expected-warning {{macro name is a reserved identifier}} */
#define _Bool _Bool                    /* expected-warning {{macro name is a reserved identifier}} */
#define __has_c_attribute __has_c_attribute /* expected-warning {{macro name is a reserved identifier}}
                                               expected-warning {{redefining builtin macro}}
                                             */
#define __restrict__ __restrict__      /* expected-warning {{macro name is a reserved identifier}} */

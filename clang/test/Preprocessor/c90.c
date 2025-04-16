/* RUN: %clang_cc1 %s -std=c89 -Eonly -DTEST_THIS_TOO -verify -pedantic-errors
 * RUN: %clang_cc1 %s -std=c89 -E | FileCheck %s
 */

/* PR3919 */

#define foo`bar   /* expected-error {{whitespace required after macro name}} */
#define foo2!bar  /* expected-warning {{whitespace recommended after macro name}} */

#define foo3$bar  /* expected-error {{whitespace required after macro name}}
                     expected-warning {{'$' in identifier; did you mean to enable '-fdollars-in-identifiers'?}}
                   */

#define test$     /* expected-error {{whitespace required after macro name}}
                     expected-warning {{'$' in identifier; did you mean to enable '-fdollars-in-identifiers'?}}
                   */
#ifdef TEST_THIS_TOO
#define $         /* expected-error {{macro name must be an identifier}}
                     expected-warning {{'$' in identifier; did you mean to enable '-fdollars-in-identifiers'?}}
                   */
#endif

/* CHECK-NOT: this comment should be missing
 * CHECK: {{^}}// this comment should be present{{$}}
 */
// this comment should be present

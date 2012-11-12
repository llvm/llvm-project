/* RUN: %clang_cc1 %s -std=c89 -Eonly -verify -pedantic-errors 
 */

/* PR3919 */

#define foo`bar   /* expected-error {{whitespace required after macro name}} */
#define foo2!bar  /* expected-warning {{whitespace recommended after macro name}} */

#define foo3$bar  /* expected-error {{'$' in identifier}} */


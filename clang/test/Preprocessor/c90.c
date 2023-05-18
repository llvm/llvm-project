/* RUN: %clang_cc1 %s -std=c89 -Eonly -verify -pedantic-errors 
 * RUN: %clang_cc1 %s -std=c89 -E | FileCheck %s
 */

/* PR3919 */

#define foo`bar   /* expected-error {{whitespace required after macro name}} */
#define foo2!bar  /* expected-warning {{whitespace recommended after macro name}} */

#define foo3$bar  /* expected-error {{'$' in identifier}} */

/* CHECK-NOT: this comment should be missing
 * CHECK: {{^}}// this comment should be present{{$}}
 */
// this comment should be present

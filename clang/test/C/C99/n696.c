// RUN: %clang_cc1 -triple x86_64 -verify %s

/* WG14 N696: yes
 * Standard pragmas - improved wording
 *
 * NB: this also covers N631 which changed these features into pragmas rather
 * than macros.
 */

// Verify that we do not expand macros in STDC pragmas. If we expanded them,
// this code would issue diagnostics.
#define ON 12
#pragma STDC FENV_ACCESS ON
#pragma STDC CX_LIMITED_RANGE ON
#pragma STDC FP_CONTRACT ON

// If we expanded macros, this code would not issue diagnostics.
#define BLERP OFF
#pragma STDC FENV_ACCESS BLERP      // expected-warning {{expected 'ON' or 'OFF' or 'DEFAULT' in pragma}}
#pragma STDC CX_LIMITED_RANGE BLERP // expected-warning {{expected 'ON' or 'OFF' or 'DEFAULT' in pragma}}
#pragma STDC FP_CONTRACT BLERP      // expected-warning {{expected 'ON' or 'OFF' or 'DEFAULT' in pragma}}


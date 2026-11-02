// RUN: %clang_cc1 -triple=x86_64-pc-linux -verify -std=c2y -Wall -pedantic %s

/* WG14 N3842: Yes
 * Standard pragmas and headers
 *
 * The standard showed #include <complex.h> before the pragmas and that
 * suggested these require header support. This paper clarified that you do not
 * need to include any headers to use standard pragmas that have component
 * which look like macros. e.g., ON, OFF, DEFAULT, CX_LIMITED_RANGE, etc.
 *
 * We claim full support because none of the pragmas are giving errors about
 * use of unknown identifiers. However, Clang doesn't currently support
 * rounding mode pragmas or DFP, so some diagnostics are expected.
 */

#pragma STDC CX_LIMITED_RANGE ON
#pragma STDC CX_LIMITED_RANGE OFF
#pragma STDC CX_LIMITED_RANGE DEFAULT

#pragma STDC FENV_ACCESS ON
#pragma STDC FENV_ACCESS OFF
#pragma STDC FENV_ACCESS DEFAULT

// Clang does not support rounding modes at all currently, so these are
// expected to warn as being unsupported.
#pragma STDC FENV_ROUND FE_DOWNWARD          // expected-warning {{pragma STDC FENV_ROUND is not supported}}
#pragma STDC FENV_ROUND FE_TONEAREST         // expected-warning {{pragma STDC FENV_ROUND is not supported}}
#pragma STDC FENV_ROUND FE_TONEARESTFROMZERO // expected-warning {{pragma STDC FENV_ROUND is not supported}}
#pragma STDC FENV_ROUND FE_TOWARDZERO        // expected-warning {{pragma STDC FENV_ROUND is not supported}}
#pragma STDC FENV_ROUND FE_UPWARD            // expected-warning {{pragma STDC FENV_ROUND is not supported}}
#pragma STDC FENV_ROUND FE_DYNAMIC           // expected-warning {{pragma STDC FENV_ROUND is not supported}}

// Clang does not support decimal floating point yet, so these are expected to
// warn as unknown pragmas.
#pragma STDC FENV_DEC_ROUND FE_DEC_DOWNWARD          // expected-warning {{unknown pragma in STDC namespace}}
#pragma STDC FENV_DEC_ROUND FE_DEC_TONEAREST         // expected-warning {{unknown pragma in STDC namespace}}
#pragma STDC FENV_DEC_ROUND FE_DEC_TONEARESTFROMZERO // expected-warning {{unknown pragma in STDC namespace}}
#pragma STDC FENV_DEC_ROUND FE_DEC_TOWARDZERO        // expected-warning {{unknown pragma in STDC namespace}}
#pragma STDC FENV_DEC_ROUND FE_DEC_UPWARD            // expected-warning {{unknown pragma in STDC namespace}}
#pragma STDC FENV_DEC_ROUND FE_DEC_DYNAMIC           // expected-warning {{unknown pragma in STDC namespace}}

#pragma STDC FP_CONTRACT ON
#pragma STDC FP_CONTRACT OFF
#pragma STDC FP_CONTRACT DEFAULT


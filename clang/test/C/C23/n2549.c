// RUN: %clang_cc1 -verify -std=c23 %s
// RUN: %clang_cc1 -verify=pedantic -std=c17 -pedantic %s
// RUN: %clang_cc1 -verify=compat -std=c23 -Wpre-c23-compat %s

// expected-no-diagnostics

/* WG14 N2549: Clang 9
 * Binary literals
 */

int i = 0b01; /* pedantic-warning {{binary integer literals are a C23 extension}}
                 compat-warning {{binary integer literals are incompatible with C standards before C23}}
               */


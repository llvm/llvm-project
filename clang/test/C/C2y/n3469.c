// RUN: %clang_cc1 -fsyntax-only -std=c2y -verify -ffreestanding %s

/* WG14 N3469: Clang 21
 * The Big Array Size Survey
 *
 * This renames _Lengthof to _Countof and introduces the stdcountof.h header.
 */

void test() {
  (void)_Countof(int[12]); // Ok
  (void)_Lengthof(int[12]); // expected-error {{use of undeclared identifier '_Lengthof'}} \
                               expected-error {{expected expression}}
}

#ifdef countof
#error "why is countof defined as a macro?"
#endif

#include <stdcountof.h>

#ifndef countof
#error "why is countof not defined as a macro?"
#endif

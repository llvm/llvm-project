// RUN: %clang_cc1 -verify=okay -std=c11 -ffreestanding %s
// RUN: %clang_cc1 -verify -std=c17 -ffreestanding %s
// RUN: %clang_cc1 -verify -std=c2x -ffreestanding %s

/* WG14 N2886: yes
 * Remove ATOMIC_VAR_INIT v2
 */

/* okay-no-diagnostics */
#include <stdatomic.h>

_Atomic int a = ATOMIC_VAR_INIT(0); /* #diag */
#if __STDC_VERSION__ <= 201710L
/* expected-warning@#diag {{macro 'ATOMIC_VAR_INIT' has been marked as deprecated}}
   expected-note@stdatomic.h:* {{macro marked 'deprecated' here}}
*/
#else
/* expected-error@#diag {{use of undeclared identifier 'ATOMIC_VAR_INIT'}} */
#endif


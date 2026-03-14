/* RUN: %clang_cc1 -std=c89 -E -verify %s | FileCheck %s
   RUN: %clang_cc1 -std=c99 -E -verify %s | FileCheck %s
   RUN: %clang_cc1 -std=c11 -E -verify %s | FileCheck %s
   RUN: %clang_cc1 -std=c17 -E -verify %s | FileCheck %s
   RUN: %clang_cc1 -std=c2x -E -verify %s | FileCheck %s
 */

/* expected-no-diagnostics */

/* WG14 DR259: yes
 * Macro invocations with no arguments
 */
#define m0() replacement
#define m1(x) begin x end

m0() m1()

/*
CHECK: replacement begin end
*/


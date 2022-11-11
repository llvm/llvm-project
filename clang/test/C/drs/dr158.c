/* RUN: %clang_cc1 -std=c89 -pedantic -Wno-c11-extensions -verify -emit-llvm -o -  %s | FileCheck %s
   RUN: %clang_cc1 -std=c99 -pedantic -Wno-c11-extensions -verify -emit-llvm -o -  %s | FileCheck %s
   RUN: %clang_cc1 -std=c11 -pedantic -verify -emit-llvm -o -  %s | FileCheck %s
   RUN: %clang_cc1 -std=c17 -pedantic -verify -emit-llvm -o -  %s | FileCheck %s
   RUN: %clang_cc1 -std=c2x -pedantic -verify -emit-llvm -o -  %s | FileCheck %s
 */

/* expected-no-diagnostics */

/* WG14 DR158: yes
 * Null pointer conversions
 */
void dr158(void) {
  int Val = (void *)0 == (int *)0;
  /* CHECK: %[[VAL:.+]] = alloca i32
     CHECK: store i32 1, ptr %[[VAL]]
   */

  (void)_Generic((int *)0, int * : 1); /* picks correct association */
  (void)_Generic((1 ? 0 : (int *)0), int * : 1); /* picks correct association */
}


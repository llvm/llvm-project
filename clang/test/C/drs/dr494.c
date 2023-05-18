/* RUN: %clang_cc1 -std=c89 %s -emit-llvm -o - | FileCheck %s
   RUN: %clang_cc1 -std=c99 %s -emit-llvm -o - | FileCheck %s
   RUN: %clang_cc1 -std=c11 %s -emit-llvm -o - | FileCheck %s
   RUN: %clang_cc1 -std=c17 %s -emit-llvm -o - | FileCheck %s
   RUN: %clang_cc1 -std=c2x %s -emit-llvm -o - | FileCheck %s
 */

/* WG14 DR494: yes
 * Part 1: Alignment specifier expression evaluation
 */
void dr494(void) {
  int i = 12;
  int j = _Alignof(int [++i]);
  int k = sizeof(int [++i]);
  /* Check that we store a straight value for i and j, but have to calculate a
   * value for storing into k. That's because sizeof() needs to execute code to
   * get the correct value from a VLA, but _Alignof is not allowed to execute
   * the VLA extent at runtime.
   */
/* CHECK: %[[I:.+]] = alloca i32
   CHECK: %[[J:.+]] = alloca i32
   CHECK: %[[K:.+]] = alloca i32
   CHECK: store i32 12, ptr %[[I]]
   CHECK: store i32 4, ptr %[[J]]
   CHECK: %[[ZERO:.+]] = load i32, ptr %[[I]]
   CHECK: %[[INC:.+]] = add nsw i32 %[[ZERO]], 1
   CHECK: store i32 %[[INC]], ptr %[[I]]
 */
}


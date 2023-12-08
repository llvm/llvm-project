/* RUN: %clang_cc1 -std=c89 -emit-llvm -o - %s | FileCheck %s
   RUN: %clang_cc1 -std=c99 -emit-llvm -o - %s | FileCheck %s
   RUN: %clang_cc1 -std=c11 -emit-llvm -o - %s | FileCheck %s
   RUN: %clang_cc1 -std=c17 -emit-llvm -o - %s | FileCheck %s
   RUN: %clang_cc1 -std=c2x -emit-llvm -o - %s | FileCheck %s
 */

/* WG14 DR094: yes
 * Are constraints on function return the same as assignment?
 */

float func(void) { return 1.0f; }
void other_func(void) {
  int i;
  float f;

  /* Test that there's been a conversion from float to int. */
  i = func();
  // CHECK: %call = call float @func()
  // CHECK-NEXT: %conv = fptosi float %call to i32
  // CHECK-NEXT: store i32 %conv, ptr %i, align 4

  /* Test that the conversion looks the same as an assignment. */
  i = f;
  // CHECK: %0 = load float, ptr %f, align 4
  // CHECK-NEXT: %conv1 = fptosi float %0 to i32
  // CHECK-NEXT: store i32 %conv1, ptr %i, align 4
}


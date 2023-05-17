/* RUN: %clang_cc1 -std=c89 -pedantic -verify -emit-llvm -o -  %s | FileCheck %s
   RUN: %clang_cc1 -std=c99 -pedantic -verify -emit-llvm -o -  %s | FileCheck %s
   RUN: %clang_cc1 -std=c11 -pedantic -verify -emit-llvm -o -  %s | FileCheck %s
   RUN: %clang_cc1 -std=c17 -pedantic -verify -emit-llvm -o -  %s | FileCheck %s
   RUN: %clang_cc1 -std=c2x -pedantic -verify -emit-llvm -o -  %s | FileCheck %s
 */

/* expected-no-diagnostics */

/* WG14 DR268: yes
 * Jumps into iteration statements
 */
void foo(void);
void dr268(void) {
  int i = 5;
  goto goto_target;

  for (i = 0; i < 10; ++i) {
    if (i > 2) ++i;
goto_target:
    foo();
  }

  /* Ensure that the goto jumps into the middle of the for loop body, and that
   * the initialization and controlling expression are not evaluated on the
   * first pass through.
   */
  /* Get us to the right function.
     CHECK-LABEL: define{{.*}} void @dr268() {{.*}} {

     First is the initialization and goto.
     CHECK: store i32 5
     CHECK-NEXT: br label %[[GOTO_TARGET:.+]]

     Then comes the initialization of the for loop variable.
     CHECK: store i32 0
     CHECK-NEXT: br label %[[FOR_COND:.+]]

     Then comes the for loop condition check label followed eventually by the
     for loop body label.
     CHECK: [[FOR_COND]]:
     CHECK: {{.+}} = icmp slt i32 {{.+}}, 10
     CHECK: [[FOR_BODY:.+]]:
     CHECK: {{.+}} = icmp sgt i32 {{.+}}, 2

     Then comes the then branch of the if statement.
     CHECK: %[[I:.+]] = load i32,
     CHECK-NEXT: %[[INC:.+]] = add nsw i32 %[[I]], 1
     CHECK-NEXT: store i32 %[[INC]],

     Eventually, we get to the goto label and its call
     CHECK: [[GOTO_TARGET]]:
     CHECK-NEXT: call void @foo()
     CHECK-NEXT: br label %[[FOR_INC:.+]]

     CHECK: [[FOR_INC]]:
     CHECK-NEXT: %[[I2:.+]] = load i32,
     CHECK-NEXT: %[[INC2:.+]] = add nsw i32 %[[I2]], 1
     CHECK-NEXT: store i32 %[[INC2]],
     CHECK-NEXT: br label %[[FOR_COND]]
   */
}


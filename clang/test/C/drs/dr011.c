/* RUN: %clang_cc1 -std=c89 -emit-llvm -o - %s | FileCheck %s
   RUN: %clang_cc1 -std=c99 -emit-llvm -o - %s | FileCheck %s
   RUN: %clang_cc1 -std=c11 -emit-llvm -o - %s | FileCheck %s
   RUN: %clang_cc1 -std=c17 -emit-llvm -o - %s | FileCheck %s
   RUN: %clang_cc1 -std=c2x -emit-llvm -o - %s | FileCheck %s
 */

/* WG14 DR011: yes
 * Merging of declarations for linked identifier
 *
 * Note, more of this DR is tested in dr0xx.c
 */

int i[10];
int j[];

// CHECK: @i = {{.*}}global [10 x i32] zeroinitializer
// CHECK-NEXT: @j = {{.*}}global [1 x i32] zeroinitializer

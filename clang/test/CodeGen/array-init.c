// RUN: %clang_cc1 %s -O0 -triple x86_64-unknown-linux-gnu -emit-llvm -o - | FileCheck -check-prefix=CHECK-NO-MERGE-CONSTANTS %s
// RUN: %clang_cc1 %s -O0 -triple x86_64-unknown-linux-gnu -fmerge-all-constants -emit-llvm -o - | FileCheck -check-prefix=CHECK-MERGE-CONSTANTS %s

// CHECK-NO-MERGE-CONSTANTS: @{{.*}}.a1 = private unnamed_addr constant [5 x i32] [i32 0, i32 1, i32 2, i32 0, i32 0]

// CHECK-MERGE-CONSTANTS: @{{.*}}.a1 = internal constant [5 x i32] [i32 0, i32 1, i32 2, i32 0, i32 0]
// CHECK-MERGE-CONSTANTS: @{{.*}}.a2 = internal constant [5 x i32] zeroinitializer
// CHECK-MERGE-CONSTANTS: @{{.*}}.a3 = internal constant [5 x i32] zeroinitializer

void testConstArrayInits(void)
{
  const int a1[5] = {0,1,2};
  const int a2[5] = {0,0,0};
  const int a3[5] = {0};
}

// RUN: %clang_cc1 -triple x86_64-apple-macosx10.11.0 -x c %s -o - -emit-llvm -fprofile-instrument=clang | FileCheck %s

// PR32019: Clang can lower some ternary operator expressions to select
// instructions. Make sure we only increment the profile counter for the
// condition when the condition evaluates to true.
// CHECK-LABEL: define{{.*}} i32 @f1
int f1(int x) {
  // CHECK: [[TOBOOL:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-NEXT: [[STEP:%.*]] = zext i1 [[TOBOOL]] to i64
  // CHECK-NEXT: [[COUNTER:%.*]] = load i64, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @__profc_f1, i32 0, i32 1)
  // CHECK-NEXT: add i64 [[COUNTER]], [[STEP]]
  // CHECK: [[COND:%.*]] = select i1 [[TOBOOL]], i32 0, i32 1
  return x ? 0 : 1;
// CHECK: ret i32 [[COND]]
}

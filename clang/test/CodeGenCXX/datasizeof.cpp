// RUN: %clang_cc1 -triple x86_64-unknown-gnu-linux -emit-llvm %s -o - | FileCheck %s

// CHECK-LABEL: define dso_local noundef i32 @_Z4testi(
// CHECK-SAME: i32 noundef [[I:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[I_ADDR:%.*]] = alloca i32, align 4
// CHECK-NEXT:    store i32 [[I]], ptr [[I_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[I_ADDR]], align 4
// CHECK-NEXT:    [[INC:%.*]] = add nsw i32 [[TMP0]], 1
// CHECK-NEXT:    store i32 [[INC]], ptr [[I_ADDR]], align 4
// CHECK-NEXT:    [[TMP1:%.*]] = zext i32 [[TMP0]] to i64
// CHECK-NEXT:    [[TMP2:%.*]] = mul nuw i64 4, [[TMP1]]
// CHECK-NEXT:    [[TMP3:%.*]] = load i32, ptr [[I_ADDR]], align 4
// CHECK-NEXT:    ret i32 [[TMP3]]
//
int test(int i) {
  (void)__datasizeof(int[i++]);
  return i;
}

// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

int test__builtin_clrsb(int x) {
// CHECK-LABEL: test__builtin_clrsb
// CHECK: [[C:%.*]] = icmp slt i32 [[X:%.*]], 0
// CHECK-NEXT: [[INV:%.*]] = xor i32 [[X]], -1
// CHECK-NEXT: [[SEL:%.*]] = select i1 [[C]], i32 [[INV]], i32 [[X]]
// CHECK-NEXT: [[CTLZ:%.*]] = call i32 @llvm.ctlz.i32(i32 [[SEL]], i1 false)
// CHECK-NEXT: [[SUB:%.*]] = sub i32 [[CTLZ]], 1
  return __builtin_clrsb(x);
}

int test__builtin_clrsbll(long long x) {
// CHECK-LABEL: test__builtin_clrsbll
// CHECK: [[C:%.*]] = icmp slt i64 [[X:%.*]], 0
// CHECK-NEXT: [[INV:%.*]] = xor i64 [[X]], -1
// CHECK-NEXT: [[SEL:%.*]] = select i1 [[C]], i64 [[INV]], i64 [[X]]
// CHECK-NEXT: [[CTLZ:%.*]] = call i64 @llvm.ctlz.i64(i64 [[SEL]], i1 false)
// CHECK-NEXT: [[SUB:%.*]] = sub i64 [[CTLZ]], 1
// CHECK-NEXT: trunc i64 [[SUB]] to i32
  return __builtin_clrsbll(x);
}

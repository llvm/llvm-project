// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

int absi(int x) {
// CHECK-LABEL: @absi(
// CHECK:   [[NEG:%.*]] = sub nsw i32 0, [[X:%.*]]
// CHECK:   [[CMP:%.*]] = icmp slt i32 [[X]], 0
// CHECK:   [[SEL:%.*]] = select i1 [[CMP]], i32 [[NEG]], i32 [[X]]
//
  return __builtin_abs(x);
}

long absl(long x) {
// CHECK-LABEL: @absl(
// CHECK:   [[NEG:%.*]] = sub nsw i64 0, [[X:%.*]]
// CHECK:   [[CMP:%.*]] = icmp slt i64 [[X]], 0
// CHECK:   [[SEL:%.*]] = select i1 [[CMP]], i64 [[NEG]], i64 [[X]]
//
  return __builtin_labs(x);
}

long long absll(long long x) {
// CHECK-LABEL: @absll(
// CHECK:   [[NEG:%.*]] = sub nsw i64 0, [[X:%.*]]
// CHECK:   [[CMP:%.*]] = icmp slt i64 [[X]], 0
// CHECK:   [[SEL:%.*]] = select i1 [[CMP]], i64 [[NEG]], i64 [[X]]
//
  return __builtin_llabs(x);
}


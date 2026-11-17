// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c23 -emit-llvm -o - %s | FileCheck %s

// C23 6.7.3.3p16: conversion to an enumeration with a fixed underlying type
// has the semantics of conversion to that type.  For an enumeration whose
// underlying type is bool the value must therefore be compared against zero,
// not truncated to its low bit.

enum Bool : bool { BF, BT };

// CHECK-LABEL: define {{.*}} i32 @from_int(
// CHECK:         [[CMP:%.*]] = icmp ne i32 {{%.*}}, 0
// CHECK-NEXT:    [[VAL:%.*]] = zext i1 [[CMP]] to i8
// CHECK-NEXT:    store i8 [[VAL]]
int from_int(int x) {
  enum Bool b = x;
  return b;
}

// CHECK-LABEL: define {{.*}} i32 @from_double(
// CHECK:         [[CMP:%.*]] = fcmp une double {{%.*}}, 0.000000e+00
// CHECK-NEXT:    [[VAL:%.*]] = zext i1 [[CMP]] to i8
// CHECK-NEXT:    store i8 [[VAL]]
int from_double(double x) {
  enum Bool b = x;
  return b;
}

// CHECK-LABEL: define {{.*}} i32 @from_float(
// CHECK:         [[CMP:%.*]] = fcmp une float {{%.*}}, 0.000000e+00
int from_float(float x) {
  return (enum Bool)x;
}

// CHECK-LABEL: define {{.*}} i32 @assign(
// CHECK:         [[CMP:%.*]] = icmp ne i32 {{%.*}}, 0
int assign(int x) {
  enum Bool b = BF;
  b = x;
  return b;
}

// Increment of a bool enumeration is always true, like increment of _Bool.

// CHECK-LABEL: define {{.*}} i32 @post_inc(
// CHECK:         store i8 1, ptr %b
int post_inc(int x) {
  enum Bool b = x;
  b++;
  return b;
}

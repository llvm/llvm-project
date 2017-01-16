// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc-apple-macosx10.4.0 -emit-llvm -o - %s -O2 -disable-llvm-passes | FileCheck %s

int boolsize = sizeof(_Bool);
// CHECK: boolsize = global i32 4, align 4

void f(_Bool *x, _Bool *y) {
  *x = *y;
}

// CHECK-LABEL: define void @f(
// CHECK: [[FROMMEM:%.*]] = load i32, i32* %
// CHECK: [[BOOLVAL:%.*]] = trunc i32 [[FROMMEM]] to i1
// CHECK: [[TOMEM:%.*]] = zext i1 [[BOOLVAL]] to i32
// CHECK: store i32 [[TOMEM]]
// CHECK: ret void

// CHECK:  i32 0, i32 2}

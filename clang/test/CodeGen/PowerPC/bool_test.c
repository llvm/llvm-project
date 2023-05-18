// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc-unknown-linux-gnu -emit-llvm -o - %s -O2 -disable-llvm-passes | FileCheck %s

int boolsize = sizeof(_Bool);
// CHECK: boolsize ={{.*}} global i32 1, align 4

void f(_Bool *x, _Bool *y) {
  *x = *y;
}

// CHECK-LABEL: define{{.*}} void @f(
// CHECK: [[FROMMEM:%.*]] = load i8, ptr %
// CHECK: [[BOOLVAL:%.*]] = trunc i8 [[FROMMEM]] to i1
// CHECK: [[TOMEM:%.*]] = zext i1 [[BOOLVAL]] to i8
// CHECK: store i8 [[TOMEM]]
// CHECK: ret void

// CHECK:  i8 0, i8 2}

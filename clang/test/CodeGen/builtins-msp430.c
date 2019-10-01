// REQUIRES: msp430-registered-target
// RUN: %clang_cc1 -triple msp430-unknown-unknown -emit-llvm %s -o - | FileCheck %s

int test_builtin_flt_rounds() {
  // CHECK:     [[V0:[%A-Za-z0-9.]+]] = call i32 @llvm.flt.rounds()
  // CHECK-DAG: [[V1:[%A-Za-z0-9.]+]] = trunc i32 [[V0]] to i16
  // CHECK-DAG: ret i16 [[V1]]
  return __builtin_flt_rounds();
}


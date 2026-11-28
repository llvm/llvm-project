// REQUIRES: msp430-registered-target
// RUN: %clang_cc1 -triple msp430-unknown-unknown -emit-llvm %s -o - | FileCheck %s

int test_builtin_flt_rounds() {
  // CHECK:     [[V0:[%A-Za-z0-9.]+]] = call i32 @llvm.get.rounding()
  // CHECK-DAG: [[V1:[%A-Za-z0-9.]+]] = trunc i32 [[V0]] to i16
  // CHECK-DAG: ret i16 [[V1]]
  return __builtin_flt_rounds();
}

void test_eh_return_data_regno(void) {
  volatile int res;
  // CHECK-LABEL: @test_eh_return_data_regno
  // CHECK: store volatile i16 12
  // CHECK: store volatile i16 13
  res = __builtin_eh_return_data_regno(0);
  res = __builtin_eh_return_data_regno(1);
}

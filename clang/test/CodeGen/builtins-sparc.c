// REQUIRES: sparc-registered-target
// RUN: %clang_cc1 -triple sparc-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple sparc64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

void test_eh_return_data_regno(void)
{
  volatile int res;
  res = __builtin_eh_return_data_regno(0);  // CHECK: store volatile i32 24
  res = __builtin_eh_return_data_regno(1);  // CHECK: store volatile i32 25
}

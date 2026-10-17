// RUN: %clang_cc1 -triple x86_64-pc-linux -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple msp430 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple avr -emit-llvm %s -o - | FileCheck %s

// The depth operand of these intrinsics is i32 regardless of the width of
// unsigned int on the target.

void *test_return_address(void) {
  // CHECK: call{{.*}} @llvm.returnaddress.p{{[0-9]}}(i32 0)
  return __builtin_return_address(0);
}

void *test_frame_address(void) {
  // CHECK: call{{.*}} @llvm.frameaddress.p{{[0-9]}}(i32 0)
  return __builtin_frame_address(0);
}

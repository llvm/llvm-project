// RUN: %clang_cc1  %s -emit-llvm -o - | FileCheck %s

void *test1() {
  // CHECK: call i8* @llvm.returnaddress
  return __builtin_return_address(1);
}
void *test2() {
  // CHECK: call i8* @llvm.frameaddress
  return __builtin_frame_address(0);
}

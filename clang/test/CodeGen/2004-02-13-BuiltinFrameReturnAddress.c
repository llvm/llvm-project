// RUN: %clang_cc1  %s -emit-llvm -o - | FileCheck %s

void *test1(void) {
  // CHECK: call ptr @llvm.returnaddress
  return __builtin_return_address(1);
}
void *test2(void) {
  // CHECK: call ptr @llvm.frameaddress
  return __builtin_frame_address(0);
}

// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +rtm -emit-llvm -o - | FileCheck %s


#include <immintrin.h>

unsigned int test_xbegin(void) {
  // CHECK: i32 @llvm.x86.xbegin()
  return _xbegin();
}

void
test_xend(void) {
  // CHECK: void @llvm.x86.xend()
  _xend();
}

void
test_xabort(void) {
  // CHECK: void @llvm.x86.xabort(i8 2)
  _xabort(2);
}

unsigned int test_xtest(void) {
  // CHECK: i32 @llvm.x86.xtest()
  return _xtest();
}

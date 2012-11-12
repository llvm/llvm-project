// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple armv7-apple-darwin9 -emit-llvm -w -o - %s | FileCheck %s

void clear0(void *ptr) {
  // CHECK: clear0
  // CHECK-NOT: load i8**
  __clear_cache();
}

void clear1(void *ptr) {
  // CHECK: clear1
  // CHECK: load i8**
  // CHECK-NOT: load i8**
  __clear_cache(ptr);
}

void clear2(void *ptr, void *ptr2) {
  // CHECK: clear2
  // CHECK: load i8**
  // CHECK: load i8**
  __clear_cache(ptr, ptr2);
}

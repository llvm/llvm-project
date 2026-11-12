// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple armv7-apple-darwin9 -emit-llvm -w -o - %s | FileCheck %s

void clear(void *ptr, void *ptr2) {
  // CHECK: clear
  // CHECK: load ptr, ptr
  // CHECK: load ptr, ptr
  __clear_cache(ptr, ptr2);
}

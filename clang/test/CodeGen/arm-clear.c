// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -no-opaque-pointers -triple armv7-apple-darwin9 -emit-llvm -w -o - %s | FileCheck %s

void clear(void *ptr, void *ptr2) {
  // CHECK: clear
  // CHECK: load i8*, i8**
  // CHECK: load i8*, i8**
  __clear_cache(ptr, ptr2);
}

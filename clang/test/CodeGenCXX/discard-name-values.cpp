// RUN: %clang_cc1 -emit-llvm  -triple=armv7-apple-darwin -emit-llvm -std=c++11 %s -o - -O1 | FileCheck %s
// RUN: %clang_cc1 -emit-llvm  -triple=armv7-apple-darwin -emit-llvm -std=c++11 %s -o - -O1 -discard-value-names | FileCheck %s --check-prefix=DISCARDVALUE

int foo(int bar) {
  return bar;
}

// CHECK: ret i32 %bar
// DISCARDVALUE: ret i32 %0


// RUN: %clang_cc1 %s -S -emit-llvm -triple x86_64-unknown-linux-gnu -o - | FileCheck %s

// Test annotation attributes on destructors do not crash.

struct k {
  ~k() __attribute__((annotate(""))) {}
};
void m() { k(); }

// CHECK: @llvm.global.annotations = appending global [2 x { ptr, ptr, ptr, i32, ptr }] [{

// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-arc -emit-llvm %s -o - | FileCheck %s

// rdar://problem/9224855
id make(void) __attribute__((ns_returns_retained));
void test0() {
  make();
  id x = 0;
  // CHECK: call void @objc_release(
  // CHECK: call void @objc_storeStrong(
}

// CHECK: declare extern_weak void @objc_release(
// CHECK: declare extern_weak void @objc_storeStrong(

// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-arc -emit-llvm %s -o - | FileCheck %s

// rdar://problem/9224855
id make(void) __attribute__((ns_returns_retained));
void test0() {
  make();
  id x = 0;
  // CHECK: call void @llvm.objc.release(
  // CHECK: call void @llvm.objc.storeStrong(
}

// CHECK: declare extern_weak void @llvm.objc.release(
// CHECK: declare extern_weak void @llvm.objc.storeStrong(

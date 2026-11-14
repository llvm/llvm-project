// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s %}

struct A { ~A(); };
void func() {
  return;
  static A k;
}

// Test that we did not crash, by checking whether function was created.
// CHECK-LABEL: define{{.*}} void @_Z4funcv() #0 {
// CHECK: ret void
// CHECK: }

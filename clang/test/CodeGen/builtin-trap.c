// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O0 %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O1 %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple wasm32-unknown-unknown -emit-llvm -O0 %s -o - | FileCheck %s

// CHECK-LABEL: define {{.*}}void @test_trap()
void test_trap(void) {
  // CHECK: call void @llvm.trap()
  // CHECK-NEXT: unreachable
  __builtin_trap();
}

// CHECK-LABEL: define {{.*}}void @test_debugtrap()
void test_debugtrap(void) {
  // CHECK: call void @llvm.debugtrap()
  // CHECK-NOT: unreachable
  __builtin_debugtrap();
}

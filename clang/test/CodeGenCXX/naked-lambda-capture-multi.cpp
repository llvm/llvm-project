// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s

void test_naked_lambda_capture_multi() {
  int x = 42;
  int y = 100;
  auto l = [&x, y]() __attribute__((naked)) {
    asm volatile("retq");
  };
  l();
}

// CHECK-LABEL: define {{.*}} @"_ZZ31test_naked_lambda_capture_multivENK3$_0clEv"
// CHECK-NOT: load i32
// CHECK-NOT: load ptr
// CHECK-NOT: getelementptr
// CHECK-NOT: alloca
// CHECK: call void asm sideeffect "retq"
// CHECK-NEXT: unreachable

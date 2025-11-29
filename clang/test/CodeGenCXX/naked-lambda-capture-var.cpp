// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -S %s -o - | FileCheck %s --check-prefix=ASM

void test_naked_lambda_capture_var() {
  int x = 42;
  auto l = [x]() __attribute__((naked)) {
    asm volatile("retq");
  };
  l();
}

// CHECK-LABEL: define {{.*}} @"_ZZ29test_naked_lambda_capture_varvENK3$_0clEv"
// CHECK-NOT: load i32
// CHECK-NOT: alloca
// CHECK-NOT: getelementptr
// CHECK: call void asm sideeffect "retq"

// ASM-LABEL: _ZZ29test_naked_lambda_capture_varvENK3$_0clEv:
// ASM-NOT: push
// ASM-NOT: mov
// ASM: retq

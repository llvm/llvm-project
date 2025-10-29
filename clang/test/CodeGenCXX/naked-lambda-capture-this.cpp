// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -S %s -o - | FileCheck %s --check-prefix=ASM

struct S {
  int member;
  void test_naked_lambda_capture_this() {
    auto l = [this]() __attribute__((naked)) {
      asm volatile("retq");
    };
    l();
  }
};

void test() {
  S s;
  s.test_naked_lambda_capture_this();
}

// CHECK-LABEL: define {{.*}} @_ZZN1S30test_naked_lambda_capture_thisEvENKUlvE_clEv
// CHECK-NOT: load ptr
// CHECK-NOT: getelementptr
// CHECK-NOT: alloca
// CHECK: call void asm sideeffect "retq"

// ASM-LABEL: _ZZN1S30test_naked_lambda_capture_thisEvENKUlvE_clEv:
// ASM-NOT: push
// ASM-NOT: mov
// ASM: retq

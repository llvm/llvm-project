// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -S %s -o - | FileCheck %s --check-prefix=ASM

void naked_lambda() {
  auto l = []() __attribute__((naked)) {
    asm volatile("retq");
  };
  l();
}
// CHECK: define internal void @"_ZZ12naked_lambdavENK3$_0clEv"
// CHECK-NOT: alloca
// CHECK-NOT: store
// ASM-LABEL: _ZZ12naked_lambdavENK3$_0clEv:
// ASM-NOT: push
// ASM-NOT: pop
// ASM: retq

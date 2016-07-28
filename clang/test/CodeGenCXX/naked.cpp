// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-windows -emit-llvm %s -o - | FileCheck %s

class TestNaked {
public:
  void NakedFunction();
};

__attribute__((naked)) void TestNaked::NakedFunction() {
  // CHECK-LABEL: define void @
  // CHECK: call void asm sideeffect
  asm("");
}

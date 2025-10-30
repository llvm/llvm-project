// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s

// Test that naked attribute is removed when captures are ODR-used (GCC compat)
void test_odr_used_captures() {
  int x = 42;
  int y = 6;
  auto l = [x, &y]() __attribute__((naked)) {
    asm volatile("movl %0, %%eax\n\tmovl %1, %%ebx\n\tretq" : : "r"(x), "r"(y));
  };
  l();
}

// CHECK-LABEL: define internal void @"_ZZ22test_odr_used_capturesvENK3$_0clEv"
// CHECK-NOT: naked
// CHECK: alloca
// CHECK: store


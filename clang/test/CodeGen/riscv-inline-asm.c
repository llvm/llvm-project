// RUN: %clang_cc1 -triple riscv32 -O2 -emit-llvm %s -o - \
// RUN:     | FileCheck %s
// RUN: %clang_cc1 -triple riscv64 -O2 -emit-llvm %s -o - \
// RUN:     | FileCheck %s

// Test RISC-V specific inline assembly constraints.

void test_I() {
// CHECK-LABEL: define void @test_I()
// CHECK: call void asm sideeffect "", "I"(i32 2047)
  asm volatile ("" :: "I"(2047));
// CHECK: call void asm sideeffect "", "I"(i32 -2048)
  asm volatile ("" :: "I"(-2048));
}

void test_J() {
// CHECK-LABEL: define void @test_J()
// CHECK: call void asm sideeffect "", "J"(i32 0)
  asm volatile ("" :: "J"(0));
}

void test_K() {
// CHECK-LABEL: define void @test_K()
// CHECK: call void asm sideeffect "", "K"(i32 31)
  asm volatile ("" :: "K"(31));
// CHECK: call void asm sideeffect "", "K"(i32 0)
  asm volatile ("" :: "K"(0));
}

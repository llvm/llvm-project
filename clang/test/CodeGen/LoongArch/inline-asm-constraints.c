// RUN: %clang_cc1 -triple loongarch32 -O2 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple loongarch64 -O2 -emit-llvm %s -o - | FileCheck %s

/// Test LoongArch specific inline assembly constraints.

float f;
double d;
void test_f(void) {
// CHECK-LABEL: define{{.*}} void @test_f()
// CHECK: [[FLT_ARG:%[a-zA-Z_0-9]+]] = load float, ptr @f
// CHECK: call void asm sideeffect "", "f"(float [[FLT_ARG]])
  asm volatile ("" :: "f"(f));
// CHECK: [[FLT_ARG:%[a-zA-Z_0-9]+]] = load double, ptr @d
// CHECK: call void asm sideeffect "", "f"(double [[FLT_ARG]])
  asm volatile ("" :: "f"(d));
}

void test_l(void) {
// CHECK-LABEL: define{{.*}} void @test_l()
// CHECK: call void asm sideeffect "", "l"(i32 32767)
  asm volatile ("" :: "l"(32767));
// CHECK: call void asm sideeffect "", "l"(i32 -32768)
  asm volatile ("" :: "l"(-32768));
}

void test_I(void) {
// CHECK-LABEL: define{{.*}} void @test_I()
// CHECK: call void asm sideeffect "", "I"(i32 2047)
  asm volatile ("" :: "I"(2047));
// CHECK: call void asm sideeffect "", "I"(i32 -2048)
  asm volatile ("" :: "I"(-2048));
}

void test_K(void) {
// CHECK-LABEL: define{{.*}} void @test_K()
// CHECK: call void asm sideeffect "", "K"(i32 4095)
  asm volatile ("" :: "K"(4095));
// CHECK: call void asm sideeffect "", "K"(i32 0)
  asm volatile ("" :: "K"(0));
}

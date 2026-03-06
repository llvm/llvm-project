// RUN: %clang_cc1 -emit-llvm -triple i386-linux -Wno-unknown-pragmas -frounding-math %s -o - | FileCheck %s

constexpr float func_01(float x, float y) {
  return x + y;
}

float V1 = func_01(1.0F, 0x0.000001p0F);
float V2 = 1.0F + 0x0.000001p0F;
float V3 = func_01(1.0F, 2.0F);

// CHECK: @V1 = {{.*}}global float 1.000000e+00, align 4
// CHECK: @V2 = {{.*}}global float 1.000000e+00, align 4
// CHECK: @V3 = {{.*}}global float 3.000000e+00, align 4

void test_builtin_elementwise_fma_round_upward() {
  #pragma STDC FENV_ACCESS ON
  #pragma STDC FENV_ROUND FE_UPWARD

  // CHECK: store float 0x4018000100000000, ptr %f1
  // CHECK: store float 0x4018000100000000, ptr %f2
  constexpr float f1 = __builtin_elementwise_fma(2.0F, 3.000001F, 0.000001F);
  constexpr float f2 = 2.0F * 3.000001F + 0.000001F;
  static_assert(f1 == f2);
  static_assert(f1 == 6.00000381F);
  // CHECK: store double 0x40180000C9539B89, ptr %d1
  // CHECK: store double 0x40180000C9539B89, ptr %d2
  constexpr double d1 = __builtin_elementwise_fma(2.0, 3.000001, 0.000001);
  constexpr double d2 = 2.0 * 3.000001 + 0.000001;
  static_assert(d1 == d2);
  static_assert(d1 == 6.0000030000000004);
}

void test_builtin_elementwise_fma_round_downward() {
  #pragma STDC FENV_ACCESS ON
  #pragma STDC FENV_ROUND FE_DOWNWARD

  // CHECK: store float 0x40180000C0000000, ptr %f3
  // CHECK: store float 0x40180000C0000000, ptr %f4
  constexpr float f3 = __builtin_elementwise_fma(2.0F, 3.000001F, 0.000001F);
  constexpr float f4 = 2.0F * 3.000001F + 0.000001F;
  static_assert(f3 == f4);
  // CHECK: store double 0x40180000C9539B87, ptr %d3
  // CHECK: store double 0x40180000C9539B87, ptr %d4
  constexpr double d3 = __builtin_elementwise_fma(2.0, 3.000001, 0.000001);
  constexpr double d4 = 2.0 * 3.000001 + 0.000001;
  static_assert(d3 == d4);
}

void test_builtin_elementwise_fma_round_nearest() {
  #pragma STDC FENV_ACCESS ON
  #pragma STDC FENV_ROUND FE_TONEAREST

  // CHECK: store float 0x40180000C0000000, ptr %f5
  // CHECK: store float 0x40180000C0000000, ptr %f6
  constexpr float f5 = __builtin_elementwise_fma(2.0F, 3.000001F, 0.000001F);
  constexpr float f6 = 2.0F * 3.000001F + 0.000001F;
  static_assert(f5 == f6);
  static_assert(f5 == 6.00000286F);
  // CHECK: store double 0x40180000C9539B89, ptr %d5
  // CHECK: store double 0x40180000C9539B89, ptr %d6
  constexpr double d5 = __builtin_elementwise_fma(2.0, 3.000001, 0.000001);
  constexpr double d6 = 2.0 * 3.000001 + 0.000001;
  static_assert(d5 == d6);
  static_assert(d5 == 6.0000030000000004);
}

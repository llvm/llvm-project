// RUN: %clang_cc1 -std=c23 -triple i386-linux %s -fsyntax-only -verify
// RUN: %clang_cc1 -std=c23 -emit-llvm -triple i386-linux %s -o - | FileCheck %s

// expected-no-diagnostics

// CHECK: @[[CONST:.*]] = private unnamed_addr constant [1 x float] [float 0x7FF8000000000000], align 4
// CHECK: @[[F_X:.*]] = internal global float 0x7FF8000000000000, align 4
#pragma STDC FENV_ACCESS ON
void f(void)
{
  // CHECK: %[[V:.*]] = alloca double, align 8
  // CHECK: %[[W:.*]] = alloca [1 x float], align 4
  // CHECK: %[[Y:.*]] = alloca float, align 4
  // CHECK: %[[Z:.*]] = alloca double, align 8

  // CHECK: store double 0x7FF8000000000000, ptr %[[V]], align 8
  constexpr double v = 0.0/0.0; // does not raise an exception

  // CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 %[[W]], ptr align 4 @[[CONST]], i32 4, i1 false)
  float w[] = { 0.0f/0.0f }; // raises an exception

  // F_X
  static float x = 0.0f/0.0f; // does not raise an exception

  // CHECK: %[[DIV:.*]] = call float @llvm.experimental.constrained.fdiv.f32(float 0.000000e+00, float 0.000000e+00, metadata !"round.dynamic", metadata !"fpexcept.strict")
  // CHECK: store float %[[DIV]], ptr %[[Y]], align 4
  float y = 0.0f/0.0f; // raises an exception

  // CHECK: %[[DIV1:.*]] = call double @llvm.experimental.constrained.fdiv.f64(double 0.000000e+00, double 0.000000e+00, metadata !"round.dynamic", metadata !"fpexcept.strict")
  // CHECK: store double %[[DIV1]], ptr %[[Z]], align 8
  double z = 0.0/0.0; // raises an exception
}

// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.2-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.2-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefix=NO_HALF

using hlsl::sqrt;

double sqrt_d(double x)
{
  return sqrt(x);
}

// CHECK: define noundef double @"?sqrt_d@@YANN@Z"(
// CHECK: call double @llvm.sqrt.f64(double %0)

float sqrt_f(float x)
{
  return sqrt(x);
}

// CHECK: define noundef float @"?sqrt_f@@YAMM@Z"(
// CHECK: call float @llvm.sqrt.f32(float %0)

half sqrt_h(half x)
{
  return sqrt(x);
}

// CHECK: define noundef half @"?sqrt_h@@YA$f16@$f16@@Z"(
// CHECK: call half @llvm.sqrt.f16(half %0)
// NO_HALF: define noundef float @"?sqrt_h@@YA$halff@$halff@@Z"(
// NO_HALF: call float @llvm.sqrt.f32(float %0)

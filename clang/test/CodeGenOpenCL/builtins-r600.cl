// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple r600-unknown-unknown -S -emit-llvm -o - %s | FileCheck %s

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// CHECK-LABEL: @test_rsq_f32
// CHECK: call float @llvm.r600.rsq.f32
void test_rsq_f32(global float* out, float a)
{
  *out = __builtin_amdgpu_rsqf(a);
}

// CHECK-LABEL: @test_rsq_f64
// CHECK: call double @llvm.r600.rsq.f64
void test_rsq_f64(global double* out, double a)
{
  *out = __builtin_amdgpu_rsq(a);
}

// CHECK-LABEL: @test_legacy_ldexp_f32
// CHECK: call float @llvm.AMDGPU.ldexp.f32
void test_legacy_ldexp_f32(global float* out, float a, int b)
{
  *out = __builtin_amdgpu_ldexpf(a, b);
}

// CHECK-LABEL: @test_legacy_ldexp_f64
// CHECK: call double @llvm.AMDGPU.ldexp.f64
void test_legacy_ldexp_f64(global double* out, double a, int b)
{
  *out = __builtin_amdgpu_ldexp(a, b);
}

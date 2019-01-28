// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx900 -S -emit-llvm -o - %s | FileCheck %s

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// CHECK-LABEL: test_interp_f16
// CHECK: call float @llvm.amdgcn.interp.p1.f16
// CHECK: call half @llvm.amdgcn.interp.p2.f16
// CHECK: call float @llvm.amdgcn.interp.p1.f16
// CHECK: call half @llvm.amdgcn.interp.p2.f16
void test_interp_f16(global half* out, float i, float j, int m0)
{
  float p1_0 = __builtin_amdgcn_interp_p1_f16(i, 2, 3, false, m0);
  half p2_0 = __builtin_amdgcn_interp_p2_f16(p1_0, j, 2, 3, false, m0);
  float p1_1 = __builtin_amdgcn_interp_p1_f16(i, 2, 3, true, m0);
  half p2_1 = __builtin_amdgcn_interp_p2_f16(p1_1, j, 2, 3, true, m0);
  *out = p2_0 + p2_1;
}

// CHECK-LABEL: test_interp_f32
// CHECK: call float @llvm.amdgcn.interp.p1
// CHECK: call float @llvm.amdgcn.interp.p2
void test_interp_f32(global float* out, float i, float j, int m0)
{
  float p1 = __builtin_amdgcn_interp_p1(i, 1, 4, m0);
  *out = __builtin_amdgcn_interp_p2(p1, j, 1, 4, m0);
}

// CHECK-LABEL: test_interp_mov
// CHECK: call float @llvm.amdgcn.interp.mov
void test_interp_mov(global float* out, float i, float j, int m0)
{
  *out = __builtin_amdgcn_interp_mov(2, 3, 4, m0);
}

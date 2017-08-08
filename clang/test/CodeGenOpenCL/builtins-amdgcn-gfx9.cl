// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx900 -S -emit-llvm -o - %s | FileCheck %s

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// CHECK-LABEL: @test_fmed3_f16
// CHECK: call half @llvm.amdgcn.fmed3.f16(half %a, half %b, half %c)
void test_fmed3_f16(global half* out, half a, half b, half c)
{
  *out = __builtin_amdgcn_fmed3h(a, b, c);
}

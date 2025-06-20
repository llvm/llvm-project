// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1200 -emit-llvm -o - %s | FileCheck %s

typedef unsigned int uint;

// CHECK-LABEL: @builtins_amdgcn_dl_insts
// CHECK: call float @llvm.amdgcn.dot4.f32.fp8.bf8(i32 %uiA, i32 %uiB, float %fC)
// CHECK: call float @llvm.amdgcn.dot4.f32.bf8.fp8(i32 %uiA, i32 %uiB, float %fC)
// CHECK: call float @llvm.amdgcn.dot4.f32.fp8.fp8(i32 %uiA, i32 %uiB, float %fC)
// CHECK: call float @llvm.amdgcn.dot4.f32.bf8.bf8(i32 %uiA, i32 %uiB, float %fC)

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
kernel void builtins_amdgcn_dl_insts_err(global float *fOut,
                                         uint uiA, uint uiB, float fC) {
  fOut[0] = __builtin_amdgcn_dot4_f32_fp8_bf8(uiA, uiB, fC);
  fOut[1] = __builtin_amdgcn_dot4_f32_bf8_fp8(uiA, uiB, fC);
  fOut[2] = __builtin_amdgcn_dot4_f32_fp8_fp8(uiA, uiB, fC);
  fOut[3] = __builtin_amdgcn_dot4_f32_bf8_bf8(uiA, uiB, fC);
}

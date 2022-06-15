// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1100 -S -emit-llvm -o - %s | FileCheck %s

typedef unsigned int uint;
typedef half __attribute__((ext_vector_type(2))) half2;
typedef short __attribute__((ext_vector_type(2))) short2;
typedef unsigned short __attribute__((ext_vector_type(2))) ushort2;

// CHECK-LABEL: @builtins_amdgcn_dl_insts
// CHECK: call float @llvm.amdgcn.fdot2(<2 x half> %v2hA, <2 x half> %v2hB, float %fC, i1 false)
// CHECK: call float @llvm.amdgcn.fdot2(<2 x half> %v2hA, <2 x half> %v2hB, float %fC, i1 true)
// CHECK: call half @llvm.amdgcn.fdot2.f16.f16(<2 x half> %v2hA, <2 x half> %v2hB, half %hC)
// CHECK: call i16 @llvm.amdgcn.fdot2.bf16.bf16(<2 x i16> %v2ssA, <2 x i16> %v2ssB, i16 %sC)
// CHECK: call float @llvm.amdgcn.fdot2.f32.bf16(<2 x i16> %v2ssA, <2 x i16> %v2ssB, float %fC, i1 false)
// CHECK: call float @llvm.amdgcn.fdot2.f32.bf16(<2 x i16> %v2ssA, <2 x i16> %v2ssB, float %fC, i1 true)
// CHECK: call i32 @llvm.amdgcn.sdot4(i32 %siA, i32 %siB, i32 %siC, i1 false)
// CHECK: call i32 @llvm.amdgcn.sdot4(i32 %siA, i32 %siB, i32 %siC, i1 true)
// CHECK: call i32 @llvm.amdgcn.udot4(i32 %uiA, i32 %uiB, i32 %uiC, i1 false)
// CHECK: call i32 @llvm.amdgcn.udot4(i32 %uiA, i32 %uiB, i32 %uiC, i1 true)
// CHECK: call i32 @llvm.amdgcn.sudot4(i1 true, i32 %A, i1 false, i32 %B, i32 %C, i1 false)
// CHECK: call i32 @llvm.amdgcn.sudot4(i1 false, i32 %A, i1 true, i32 %B, i32 %C, i1 true)
// CHECK: call i32 @llvm.amdgcn.sdot8(i32 %siA, i32 %siB, i32 %siC, i1 false)
// CHECK: call i32 @llvm.amdgcn.sdot8(i32 %siA, i32 %siB, i32 %siC, i1 true)
// CHECK: call i32 @llvm.amdgcn.udot8(i32 %uiA, i32 %uiB, i32 %uiC, i1 false)
// CHECK: call i32 @llvm.amdgcn.udot8(i32 %uiA, i32 %uiB, i32 %uiC, i1 true)
// CHECK: call i32 @llvm.amdgcn.sudot8(i1 false, i32 %A, i1 true, i32 %B, i32 %C, i1 false)
// CHECK: call i32 @llvm.amdgcn.sudot8(i1 true, i32 %A, i1 false, i32 %B, i32 %C, i1 true)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
kernel void builtins_amdgcn_dl_insts_err(
    global float *fOut, global int *siOut, global uint *uiOut,
    global short *sOut, global int *iOut, global half *hOut,
    half2 v2hA, half2 v2hB, float fC, half hC,
    short2 v2ssA, short2 v2ssB, short sC, int siA, int siB, int siC,
    ushort2 v2usA, ushort2 v2usB, uint uiA, uint uiB, uint uiC,
    int A, int B, int C) {
  fOut[0] = __builtin_amdgcn_fdot2(v2hA, v2hB, fC, false);
  fOut[1] = __builtin_amdgcn_fdot2(v2hA, v2hB, fC, true);

  hOut[0] = __builtin_amdgcn_fdot2_f16_f16(v2hA, v2hB, hC);

  sOut[0] = __builtin_amdgcn_fdot2_bf16_bf16(v2ssA, v2ssB, sC);

  fOut[3] = __builtin_amdgcn_fdot2_f32_bf16(v2ssA, v2ssB, fC, false);
  fOut[4] = __builtin_amdgcn_fdot2_f32_bf16(v2ssA, v2ssB, fC, true);

  siOut[2] = __builtin_amdgcn_sdot4(siA, siB, siC, false);
  siOut[3] = __builtin_amdgcn_sdot4(siA, siB, siC, true);

  uiOut[2] = __builtin_amdgcn_udot4(uiA, uiB, uiC, false);
  uiOut[3] = __builtin_amdgcn_udot4(uiA, uiB, uiC, true);

  iOut[0] = __builtin_amdgcn_sudot4(true, A, false, B, C, false);
  iOut[1] = __builtin_amdgcn_sudot4(false, A, true, B, C, true);

  siOut[4] = __builtin_amdgcn_sdot8(siA, siB, siC, false);
  siOut[5] = __builtin_amdgcn_sdot8(siA, siB, siC, true);

  uiOut[4] = __builtin_amdgcn_udot8(uiA, uiB, uiC, false);
  uiOut[5] = __builtin_amdgcn_udot8(uiA, uiB, uiC, true);

  iOut[3] = __builtin_amdgcn_sudot8(false, A, true, B, C, false);
  iOut[4] = __builtin_amdgcn_sudot8(true, A, false, B, C, true);
}

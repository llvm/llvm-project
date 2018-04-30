// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx906 -S -emit-llvm -o - %s | FileCheck %s

typedef unsigned int uint;
typedef half __attribute__((ext_vector_type(2))) half2;
typedef short __attribute__((ext_vector_type(2))) short2;
typedef unsigned short __attribute__((ext_vector_type(2))) ushort2;

// CHECK-LABEL: @builtins_amdgcn_dl_insts
// CHECK: call float @llvm.amdgcn.fdot2

// CHECK: call i32 @llvm.amdgcn.sdot2
// CHECK: call i32 @llvm.amdgcn.udot2

// CHECK: call i32 @llvm.amdgcn.sdot4
// CHECK: call i32 @llvm.amdgcn.udot4

// CHECK: call i32 @llvm.amdgcn.sdot8
// CHECK: call i32 @llvm.amdgcn.udot8
kernel void builtins_amdgcn_dl_insts(
    global float *fOut, global int *siOut, global uint *uiOut,
    half2 v2hA, half2 v2hB, float fC,
    short2 v2ssA, short2 v2ssB, int siA, int siB, int siC,
    ushort2 v2usA, ushort2 v2usB, uint uiA, uint uiB, uint uiC) {
  fOut[0] = __builtin_amdgcn_fdot2(v2hA, v2hB, fC);

  siOut[0] = __builtin_amdgcn_sdot2(v2ssA, v2ssB, siC);
  uiOut[0] = __builtin_amdgcn_udot2(v2usA, v2usB, uiC);

  siOut[1] = __builtin_amdgcn_sdot4(siA, siB, siC);
  uiOut[1] = __builtin_amdgcn_udot4(uiA, uiB, uiC);

  siOut[2] = __builtin_amdgcn_sdot8(siA, siB, siC);
  uiOut[2] = __builtin_amdgcn_udot8(uiA, uiB, uiC);
}

// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx906 -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1011 -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1012 -S -emit-llvm -o - %s | FileCheck %s

typedef unsigned int uint;
typedef half __attribute__((ext_vector_type(2))) half2;
typedef short __attribute__((ext_vector_type(2))) short2;
typedef unsigned short __attribute__((ext_vector_type(2))) ushort2;

// CHECK-LABEL: @builtins_amdgcn_dl_insts
// CHECK: call float @llvm.amdgcn.fdot2(<2 x half> %v2hA, <2 x half> %v2hB, float %fC, i1 false)
// CHECK: call float @llvm.amdgcn.fdot2(<2 x half> %v2hA, <2 x half> %v2hB, float %fC, i1 true)

// CHECK: call i32 @llvm.amdgcn.sdot2(<2 x i16> %v2ssA, <2 x i16> %v2ssB, i32 %siC, i1 false)
// CHECK: call i32 @llvm.amdgcn.sdot2(<2 x i16> %v2ssA, <2 x i16> %v2ssB, i32 %siC, i1 true)

// CHECK: call i32 @llvm.amdgcn.udot2(<2 x i16> %v2usA, <2 x i16> %v2usB, i32 %uiC, i1 false)
// CHECK: call i32 @llvm.amdgcn.udot2(<2 x i16> %v2usA, <2 x i16> %v2usB, i32 %uiC, i1 true)

// CHECK: call i32 @llvm.amdgcn.sdot4(i32 %siA, i32 %siB, i32 %siC, i1 false)
// CHECK: call i32 @llvm.amdgcn.sdot4(i32 %siA, i32 %siB, i32 %siC, i1 true)

// CHECK: call i32 @llvm.amdgcn.udot4(i32 %uiA, i32 %uiB, i32 %uiC, i1 false)
// CHECK: call i32 @llvm.amdgcn.udot4(i32 %uiA, i32 %uiB, i32 %uiC, i1 true)

// CHECK: call i32 @llvm.amdgcn.sdot8(i32 %siA, i32 %siB, i32 %siC, i1 false)
// CHECK: call i32 @llvm.amdgcn.sdot8(i32 %siA, i32 %siB, i32 %siC, i1 true)

// CHECK: call i32 @llvm.amdgcn.udot8(i32 %uiA, i32 %uiB, i32 %uiC, i1 false)
// CHECK: call i32 @llvm.amdgcn.udot8(i32 %uiA, i32 %uiB, i32 %uiC, i1 true)
kernel void builtins_amdgcn_dl_insts(
    global float *fOut, global int *siOut, global uint *uiOut,
    half2 v2hA, half2 v2hB, float fC,
    short2 v2ssA, short2 v2ssB, int siA, int siB, int siC,
    ushort2 v2usA, ushort2 v2usB, uint uiA, uint uiB, uint uiC) {
  fOut[0] = __builtin_amdgcn_fdot2(v2hA, v2hB, fC, false);
  fOut[1] = __builtin_amdgcn_fdot2(v2hA, v2hB, fC, true);

  siOut[0] = __builtin_amdgcn_sdot2(v2ssA, v2ssB, siC, false);
  siOut[1] = __builtin_amdgcn_sdot2(v2ssA, v2ssB, siC, true);

  uiOut[0] = __builtin_amdgcn_udot2(v2usA, v2usB, uiC, false);
  uiOut[1] = __builtin_amdgcn_udot2(v2usA, v2usB, uiC, true);

  siOut[2] = __builtin_amdgcn_sdot4(siA, siB, siC, false);
  siOut[3] = __builtin_amdgcn_sdot4(siA, siB, siC, true);

  uiOut[2] = __builtin_amdgcn_udot4(uiA, uiB, uiC, false);
  uiOut[3] = __builtin_amdgcn_udot4(uiA, uiB, uiC, true);

  siOut[4] = __builtin_amdgcn_sdot8(siA, siB, siC, false);
  siOut[5] = __builtin_amdgcn_sdot8(siA, siB, siC, true);

  uiOut[4] = __builtin_amdgcn_udot8(uiA, uiB, uiC, false);
  uiOut[5] = __builtin_amdgcn_udot8(uiA, uiB, uiC, true);
}

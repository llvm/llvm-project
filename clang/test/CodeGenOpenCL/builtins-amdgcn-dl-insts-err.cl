// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx900 -verify -S -emit-llvm -o - %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1010 -verify -S -emit-llvm -o - %s

typedef unsigned int uint;
typedef half __attribute__((ext_vector_type(2))) half2;
typedef short __attribute__((ext_vector_type(2))) short2;
typedef unsigned short __attribute__((ext_vector_type(2))) ushort2;

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
kernel void builtins_amdgcn_dl_insts_err(
    global float *fOut, global int *siOut, global uint *uiOut,
    global short *sOut, global int *iOut, global half *hOut,
    half2 v2hA, half2 v2hB, float fC, half hC,
    short2 v2ssA, short2 v2ssB, short sC, int siA, int siB, int siC,
    ushort2 v2usA, ushort2 v2usB, uint uiA, uint uiB, uint uiC,
    int A, int B, int C) {
  fOut[0] = __builtin_amdgcn_fdot2(v2hA, v2hB, fC, false);          // expected-error {{'__builtin_amdgcn_fdot2' needs target feature dot10-insts}}
  fOut[1] = __builtin_amdgcn_fdot2(v2hA, v2hB, fC, true);           // expected-error {{'__builtin_amdgcn_fdot2' needs target feature dot10-insts}}

  hOut[0] = __builtin_amdgcn_fdot2_f16_f16(v2hA, v2hB, hC);           // expected-error {{'__builtin_amdgcn_fdot2_f16_f16' needs target feature dot9-insts}}

  sOut[0] = __builtin_amdgcn_fdot2_bf16_bf16(v2ssA, v2ssB, sC);       // expected-error {{'__builtin_amdgcn_fdot2_bf16_bf16' needs target feature dot9-insts}}

  fOut[3] = __builtin_amdgcn_fdot2_f32_bf16(v2ssA, v2ssB, fC, false); // expected-error {{'__builtin_amdgcn_fdot2_f32_bf16' needs target feature dot9-insts}}
  fOut[4] = __builtin_amdgcn_fdot2_f32_bf16(v2ssA, v2ssB, fC, true);  // expected-error {{'__builtin_amdgcn_fdot2_f32_bf16' needs target feature dot9-insts}}

  siOut[0] = __builtin_amdgcn_sdot2(v2ssA, v2ssB, siC, false);      // expected-error {{'__builtin_amdgcn_sdot2' needs target feature dot2-insts}}
  siOut[1] = __builtin_amdgcn_sdot2(v2ssA, v2ssB, siC, true);       // expected-error {{'__builtin_amdgcn_sdot2' needs target feature dot2-insts}}

  uiOut[0] = __builtin_amdgcn_udot2(v2usA, v2usB, uiC, false);      // expected-error {{'__builtin_amdgcn_udot2' needs target feature dot2-insts}}
  uiOut[1] = __builtin_amdgcn_udot2(v2usA, v2usB, uiC, true);       // expected-error {{'__builtin_amdgcn_udot2' needs target feature dot2-insts}}

  siOut[2] = __builtin_amdgcn_sdot4(siA, siB, siC, false);          // expected-error {{'__builtin_amdgcn_sdot4' needs target feature dot1-insts}}
  siOut[3] = __builtin_amdgcn_sdot4(siA, siB, siC, true);           // expected-error {{'__builtin_amdgcn_sdot4' needs target feature dot1-insts}}

  uiOut[2] = __builtin_amdgcn_udot4(uiA, uiB, uiC, false);          // expected-error {{'__builtin_amdgcn_udot4' needs target feature dot7-insts}}
  uiOut[3] = __builtin_amdgcn_udot4(uiA, uiB, uiC, true);           // expected-error {{'__builtin_amdgcn_udot4' needs target feature dot7-insts}}

  iOut[0] = __builtin_amdgcn_sudot4(true, A, false, B, C, false);   // expected-error {{'__builtin_amdgcn_sudot4' needs target feature dot8-insts}}
  iOut[1] = __builtin_amdgcn_sudot4(false, A, true, B, C, true);    // expected-error {{'__builtin_amdgcn_sudot4' needs target feature dot8-insts}}

  siOut[4] = __builtin_amdgcn_sdot8(siA, siB, siC, false);          // expected-error {{'__builtin_amdgcn_sdot8' needs target feature dot1-insts}}
  siOut[5] = __builtin_amdgcn_sdot8(siA, siB, siC, true);           // expected-error {{'__builtin_amdgcn_sdot8' needs target feature dot1-insts}}

  uiOut[4] = __builtin_amdgcn_udot8(uiA, uiB, uiC, false);          // expected-error {{'__builtin_amdgcn_udot8' needs target feature dot7-insts}}
  uiOut[5] = __builtin_amdgcn_udot8(uiA, uiB, uiC, true);           // expected-error {{'__builtin_amdgcn_udot8' needs target feature dot7-insts}}

  iOut[3] = __builtin_amdgcn_sudot8(false, A, true, B, C, false);    // expected-error {{'__builtin_amdgcn_sudot8' needs target feature dot8-insts}}
  iOut[4] = __builtin_amdgcn_sudot8(true, A, false, B, C, true);     // expected-error {{'__builtin_amdgcn_sudot8' needs target feature dot8-insts}}
}

// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx906 -verify -emit-llvm -o - %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1010 -verify -emit-llvm -o - %s

typedef unsigned int uint;
typedef half __attribute__((ext_vector_type(2))) half2;
typedef short __attribute__((ext_vector_type(2))) short2;
typedef unsigned short __attribute__((ext_vector_type(2))) ushort2;

kernel void builtins_amdgcn_dl_insts_err(
    global float *fOut, global int *siOut, global uint *uiOut,
    half2 v2hA, half2 v2hB, float fC,
    short2 v2ssA, short2 v2ssB, int siA, int siB, int siC,
    ushort2 v2usA, ushort2 v2usB, uint uiA, uint uiB, uint uiC, uint isClamp) {
  fOut[0] = __builtin_amdgcn_fdot2(v2hA, v2hB, fC, isClamp == 0 ? false : true);     // expected-error {{'__builtin_amdgcn_fdot2' must be a constant integer}}

  siOut[0] = __builtin_amdgcn_sdot2(v2ssA, v2ssB, siC, isClamp == 0 ? false : true); // expected-error {{'__builtin_amdgcn_sdot2' must be a constant integer}}
  uiOut[0] = __builtin_amdgcn_udot2(v2usA, v2usB, uiC, isClamp == 0 ? false : true); // expected-error {{'__builtin_amdgcn_udot2' must be a constant integer}}

  siOut[1] = __builtin_amdgcn_sdot4(siA, siB, siC, isClamp == 0 ? false : true);     // expected-error {{'__builtin_amdgcn_sdot4' must be a constant integer}}
  uiOut[1] = __builtin_amdgcn_udot4(uiA, uiB, uiC, isClamp == 0 ? false : true);     // expected-error {{'__builtin_amdgcn_udot4' must be a constant integer}}

  siOut[2] = __builtin_amdgcn_sdot8(siA, siB, siC, isClamp == 0 ? false : true);     // expected-error {{'__builtin_amdgcn_sdot8' must be a constant integer}}
  uiOut[2] = __builtin_amdgcn_udot8(uiA, uiB, uiC, isClamp == 0 ? false : true);     // expected-error {{'__builtin_amdgcn_udot8' must be a constant integer}}
}

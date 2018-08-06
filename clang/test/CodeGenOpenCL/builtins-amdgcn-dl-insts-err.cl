// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx900 -verify -S -emit-llvm -o - %s

typedef unsigned int uint;
typedef half __attribute__((ext_vector_type(2))) half2;
typedef short __attribute__((ext_vector_type(2))) short2;
typedef unsigned short __attribute__((ext_vector_type(2))) ushort2;

kernel void builtins_amdgcn_dl_insts_err(
    global float *fOut, global int *siOut, global uint *uiOut,
    half2 v2hA, half2 v2hB, float fC,
    short2 v2ssA, short2 v2ssB, int siA, int siB, int siC,
    ushort2 v2usA, ushort2 v2usB, uint uiA, uint uiB, uint uiC) {
  fOut[0] = __builtin_amdgcn_fdot2(v2hA, v2hB, fC); // expected-error {{'__builtin_amdgcn_fdot2' needs target feature dl-insts}}

  siOut[0] = __builtin_amdgcn_sdot2(v2ssA, v2ssB, siC); // expected-error {{'__builtin_amdgcn_sdot2' needs target feature dl-insts}}
  uiOut[0] = __builtin_amdgcn_udot2(v2usA, v2usB, uiC); // expected-error {{'__builtin_amdgcn_udot2' needs target feature dl-insts}}

  siOut[1] = __builtin_amdgcn_sdot4(siA, siB, siC); // expected-error {{'__builtin_amdgcn_sdot4' needs target feature dl-insts}}
  uiOut[1] = __builtin_amdgcn_udot4(uiA, uiB, uiC); // expected-error {{'__builtin_amdgcn_udot4' needs target feature dl-insts}}

  siOut[2] = __builtin_amdgcn_sdot8(siA, siB, siC); // expected-error {{'__builtin_amdgcn_sdot8' needs target feature dl-insts}}
  uiOut[2] = __builtin_amdgcn_udot8(uiA, uiB, uiC); // expected-error {{'__builtin_amdgcn_udot8' needs target feature dl-insts}}
}

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx906 -DTEST_FMA -verify=ok -S -o /dev/null %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx906 -DTEST_BF16 -verify=nobf16 -S -o /dev/null %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx900 -DTEST_FMA -verify=nofma -S -o /dev/null %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1250 -DTEST_FMA -verify=ok -S -o /dev/null %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1250 -DTEST_BF16 -verify=ok -S -o /dev/null %s

// REQUIRES: amdgpu-registered-target

// ok-no-diagnostics

typedef unsigned int uint;

#ifdef TEST_FMA
float test_fma(uint src0, uint src1, uint src2) {
  return __builtin_amdgcn_fma_mix_f32(src0, src1, src2, 0, 0, 0); // nofma-error {{'__builtin_amdgcn_fma_mix_f32' needs target feature fma-mix-insts}}
}
#endif

#ifdef TEST_BF16
float test_bf16(uint src0, uint src1, uint src2) {
  return __builtin_amdgcn_fma_mix_f32_bf16(src0, src1, src2, 0, 0, 0); // nobf16-error {{'__builtin_amdgcn_fma_mix_f32_bf16' needs target feature fma-mix-bf16-insts}}
}
#endif

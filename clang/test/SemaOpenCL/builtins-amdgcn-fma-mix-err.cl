// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1250 -verify=selector -S -o /dev/null %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx900 -DUNSUPPORTED -verify=feature -S -o /dev/null %s

// REQUIRES: amdgpu-registered-target

typedef unsigned int uint;
struct Bad {
  int x;
};

#ifndef UNSUPPORTED
void test_fma_mix_selectors(uint src0, uint src1, uint src2, uint selector) {
  __builtin_amdgcn_fma_mix_f32(src0, src1, src2, selector, 0, 0); // selector-error {{argument to '__builtin_amdgcn_fma_mix_f32' must be a constant integer}}
  __builtin_amdgcn_fma_mix_f32(src0, src1, src2, 0, selector, 0); // selector-error {{argument to '__builtin_amdgcn_fma_mix_f32' must be a constant integer}}
  __builtin_amdgcn_fma_mix_f32(src0, src1, src2, 0, 0, selector); // selector-error {{argument to '__builtin_amdgcn_fma_mix_f32' must be a constant integer}}
  __builtin_amdgcn_fma_mix_f32(src0, src1, src2, -1, 0, 0); // selector-error {{argument value 4294967295 is outside the valid range [0, 3]}}
  __builtin_amdgcn_fma_mix_f32(src0, src1, src2, 4, 0, 0); // selector-error {{argument value 4 is outside the valid range [0, 3]}}
  __builtin_amdgcn_fma_mix_f32(src0, src1, src2, 0, 4, 0); // selector-error {{argument value 4 is outside the valid range [0, 3]}}
  __builtin_amdgcn_fma_mix_f32(src0, src1, src2, 0, 0, 4); // selector-error {{argument value 4 is outside the valid range [0, 3]}}
}

void test_fma_mix_packed_selectors(uint src0, uint src1, uint src2, uint dst,
                                   uint selector) {
  __builtin_amdgcn_fma_mixlo_f16(src0, src1, src2, dst, selector, 0, 0); // selector-error {{argument to '__builtin_amdgcn_fma_mixlo_f16' must be a constant integer}}
  __builtin_amdgcn_fma_mixlo_f16(src0, src1, src2, dst, 0, selector, 0); // selector-error {{argument to '__builtin_amdgcn_fma_mixlo_f16' must be a constant integer}}
  __builtin_amdgcn_fma_mixlo_f16(src0, src1, src2, dst, 0, 0, selector); // selector-error {{argument to '__builtin_amdgcn_fma_mixlo_f16' must be a constant integer}}
  __builtin_amdgcn_fma_mixlo_f16(src0, src1, src2, dst, 4, 0, 0); // selector-error {{argument value 4 is outside the valid range [0, 3]}}
  __builtin_amdgcn_fma_mixlo_f16(src0, src1, src2, dst, 0, 4, 0); // selector-error {{argument value 4 is outside the valid range [0, 3]}}
  __builtin_amdgcn_fma_mixlo_f16(src0, src1, src2, dst, 0, 0, 4); // selector-error {{argument value 4 is outside the valid range [0, 3]}}
}

void test_signatures(uint src0, uint src1, uint src2, uint dst,
                     struct Bad bad) {
  (void)__builtin_amdgcn_fma_mix_f32(src0, src1, src2, 0, 0); // selector-error {{too few arguments to function call, expected 6, have 5}}
  (void)__builtin_amdgcn_fma_mix_f32(src0, src1, src2, 0, 0, 0, 0); // selector-error {{too many arguments to function call, expected 6, have 7}}
  (void)__builtin_amdgcn_fma_mix_f32(bad, src1, src2, 0, 0, 0); // selector-error {{passing '__private struct Bad' to parameter of incompatible type 'unsigned int'}}
  (void)__builtin_amdgcn_fma_mixlo_f16(src0, src1, src2, dst, 0, 0); // selector-error {{too few arguments to function call, expected 7, have 6}}
  (void)__builtin_amdgcn_fma_mixlo_f16(src0, src1, src2, dst, 0, 0, 0, 0); // selector-error {{too many arguments to function call, expected 7, have 8}}
  (void)__builtin_amdgcn_fma_mixlo_f16(bad, src1, src2, dst, 0, 0, 0); // selector-error {{passing '__private struct Bad' to parameter of incompatible type 'unsigned int'}}
}
#else
void test_unsupported(uint src0, uint src1, uint src2, uint dst) {
  (void)__builtin_amdgcn_fma_mix_f32(src0, src1, src2, 0, 0, 0); // feature-error {{'__builtin_amdgcn_fma_mix_f32' needs target feature fma-mix-insts}}
  (void)__builtin_amdgcn_fma_mix_f32_bf16(src0, src1, src2, 0, 0, 0); // feature-error {{'__builtin_amdgcn_fma_mix_f32_bf16' needs target feature fma-mix-bf16-insts}}
  (void)__builtin_amdgcn_fma_mixlo_f16(src0, src1, src2, dst, 0, 0, 0); // feature-error {{'__builtin_amdgcn_fma_mixlo_f16' needs target feature fma-mix-insts}}
  (void)__builtin_amdgcn_fma_mixhi_f16(src0, src1, src2, dst, 0, 0, 0); // feature-error {{'__builtin_amdgcn_fma_mixhi_f16' needs target feature fma-mix-insts}}
  (void)__builtin_amdgcn_fma_mixlo_bf16(src0, src1, src2, dst, 0, 0, 0); // feature-error {{'__builtin_amdgcn_fma_mixlo_bf16' needs target feature fma-mix-bf16-insts}}
  (void)__builtin_amdgcn_fma_mixhi_bf16(src0, src1, src2, dst, 0, 0, 0); // feature-error {{'__builtin_amdgcn_fma_mixhi_bf16' needs target feature fma-mix-bf16-insts}}
}
#endif

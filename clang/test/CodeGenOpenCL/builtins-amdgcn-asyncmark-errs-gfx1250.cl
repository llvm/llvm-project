// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgpu12.50-amd-amdhsa -verify -S -o - %s

// The stage mask must be a compile-time constant.

void test_mask_not_constant(unsigned int m) {
  __builtin_amdgcn_asyncmark(m); // expected-error{{argument to '__builtin_amdgcn_asyncmark' must be a constant integer}}
  __builtin_amdgcn_wait_asyncmark(0, m); // expected-error{{argument to '__builtin_amdgcn_wait_asyncmark' must be a constant integer}}
}

// A mask may only name stages that exist. 2047 names all eleven of them, and
// 2048 is the bit just past the last one.

void test_mask_out_of_range() {
  __builtin_amdgcn_asyncmark(2048); // expected-error{{argument value 2048 is outside the valid range [0, 2047]}}
  __builtin_amdgcn_asyncmark(4096); // expected-error{{argument value 4096 is outside the valid range [0, 2047]}}
  __builtin_amdgcn_wait_asyncmark(0, 2048); // expected-error{{argument value 2048 is outside the valid range [0, 2047]}}
}

// Every combination of in-range bits is accepted, including bits of the
// reserved stages: leaving out a stage whose operations do not exist yet is
// harmless, and keeps masks portable as the reserved slots are filled in.

void test_mask_reserved_bits_ok() {
  __builtin_amdgcn_asyncmark(16);   // RESERVED_4
  __builtin_amdgcn_asyncmark(2000); // every reserved stage at once
  __builtin_amdgcn_asyncmark(2047); // every stage
  __builtin_amdgcn_wait_asyncmark(0, 16);
  __builtin_amdgcn_wait_asyncmark(0, 2047);
}

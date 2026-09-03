// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgpu12.50-amd-amdhsa -verify -S -o - %s

// The stage must be a compile-time constant; _Constant alone does not give a
// range, so the value is checked here rather than left to the backend.

void test_stage_not_constant(unsigned int s) {
  __builtin_amdgcn_asyncmark(s); // expected-error{{argument to '__builtin_amdgcn_asyncmark' must be a constant integer}}
  __builtin_amdgcn_wait_asyncmark(0, s); // expected-error{{argument to '__builtin_amdgcn_wait_asyncmark' must be a constant integer}}
}

// Stages above the catch-all, and the gap between the last named stage and the
// catch-all, are out of range.

void test_stage_out_of_range() {
  __builtin_amdgcn_asyncmark(11); // expected-error{{argument value 11 is outside the valid range [0, 16]}}
  __builtin_amdgcn_asyncmark(15); // expected-error{{argument value 15 is outside the valid range [0, 16]}}
  __builtin_amdgcn_asyncmark(17); // expected-error{{argument value 17 is outside the valid range [0, 16]}}
  __builtin_amdgcn_wait_asyncmark(0, 17); // expected-error{{argument value 17 is outside the valid range [0, 16]}}
}

// The reserved stages hold slots in the taxonomy for asynchronous operations
// this target does not have. Using one is an error, not a silent no-op.

void test_stage_reserved() {
  __builtin_amdgcn_asyncmark(4); // expected-error{{asyncmark stage RESERVED_4 is reserved}}
  __builtin_amdgcn_asyncmark(6); // expected-error{{asyncmark stage RESERVED_6 is reserved}}
  __builtin_amdgcn_asyncmark(7); // expected-error{{asyncmark stage RESERVED_7 is reserved}}
  __builtin_amdgcn_asyncmark(8); // expected-error{{asyncmark stage RESERVED_8 is reserved}}
  __builtin_amdgcn_asyncmark(9); // expected-error{{asyncmark stage RESERVED_9 is reserved}}
  __builtin_amdgcn_asyncmark(10); // expected-error{{asyncmark stage RESERVED_10 is reserved}}
  __builtin_amdgcn_wait_asyncmark(0, 4); // expected-error{{asyncmark stage RESERVED_4 is reserved}}
}

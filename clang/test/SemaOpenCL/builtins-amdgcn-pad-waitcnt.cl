// RUN: %clang_cc1 -triple amdgpu9.50-amd-amdhsa -cl-std=CL2.0 -fsyntax-only -verify %s
// REQUIRES: amdgpu-registered-target

void immediate_arguments(int counter, int count) {
  __builtin_amdgcn_pad_waitcnt(counter, 1); // expected-error {{must be a constant integer}}
  __builtin_amdgcn_pad_waitcnt(0, count); // expected-error {{must be a constant integer}}
  // Unsupported values are optional hints, not frontend errors.
  __builtin_amdgcn_pad_waitcnt(0, 0);
  __builtin_amdgcn_pad_waitcnt(0, -1);
  __builtin_amdgcn_pad_waitcnt(1, 1);
}

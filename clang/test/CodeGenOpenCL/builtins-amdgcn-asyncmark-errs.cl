// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx1250 -verify -S -o - %s

void test_feature() {
  __builtin_amdgcn_asyncmark(); // expected-error{{'__builtin_amdgcn_asyncmark' needs target feature vmem-to-lds-load-insts}}
  __builtin_amdgcn_wait_asyncmark(0); // expected-error{{'__builtin_amdgcn_wait_asyncmark' needs target feature vmem-to-lds-load-insts}}
}

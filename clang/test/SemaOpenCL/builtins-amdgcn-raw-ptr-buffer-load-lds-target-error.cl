// RUN: %clang_cc1 -triple amdgpu6.00-unknown-unknown -S -verify -o - %s
// RUN: %clang_cc1 -triple amdgpu7.04-unknown-unknown -S -verify -o - %s
// RUN: %clang_cc1 -triple amdgpu8.01-unknown-unknown -S -verify -o - %s
// RUN: %clang_cc1 -triple amdgpu11.00-unknown-unknown -S -verify -o - %s
// RUN: %clang_cc1 -triple amdgpu12.00-unknown-unknown -S -verify -o - %s
// REQUIRES: amdgpu-registered-target

void test_amdgcn_raw_ptr_buffer_load_lds(__amdgpu_buffer_rsrc_t rsrc, __local void* lds, int vindex, int offset, int soffset) {
  __builtin_amdgcn_raw_ptr_buffer_load_lds(rsrc, lds, 4, offset, soffset, 0, 0); //expected-error{{needs target feature vmem-to-lds-load-insts}}
  __builtin_amdgcn_struct_ptr_buffer_load_lds(rsrc, lds, 4, vindex, offset, soffset, 0, 0); //expected-error{{needs target feature vmem-to-lds-load-insts}}
}

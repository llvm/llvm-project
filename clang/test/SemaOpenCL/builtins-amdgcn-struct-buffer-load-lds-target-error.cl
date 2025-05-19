// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu tahiti -S -verify -o - %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu bonaire -S -verify -o - %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu carrizo -S -verify -o - %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1100 -S -verify -o - %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1200 -S -verify -o - %s
// REQUIRES: amdgpu-registered-target

typedef unsigned int v4u32 __attribute__((ext_vector_type(4)));

void test_amdgcn_struct_buffer_load_lds(v4u32 rsrc, __local void* lds, int index, int offset, int soffset, int x) {
  __builtin_amdgcn_struct_buffer_load_lds(rsrc, lds, 4, index, offset, soffset, 0, 0); //expected-error{{needs target feature vmem-to-lds-load-insts}}
}

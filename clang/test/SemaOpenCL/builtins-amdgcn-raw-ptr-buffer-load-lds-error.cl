// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx90a -S -verify=gfx90a,expected -o - %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx950 -S -verify=gfx950,expected  -o - %s
// REQUIRES: amdgpu-registered-target

void test_amdgcn_raw_ptr_buffer_load_lds(__amdgpu_buffer_rsrc_t rsrc, __local void* lds, int offset, int soffset, int x) {
  __builtin_amdgcn_raw_ptr_buffer_load_lds(rsrc, lds, x, offset, soffset, 0, 0); //expected-error{{argument to '__builtin_amdgcn_raw_ptr_buffer_load_lds' must be a constant integer}}
  __builtin_amdgcn_raw_ptr_buffer_load_lds(rsrc, lds, 4, offset, soffset, x, 0); //expected-error{{argument to '__builtin_amdgcn_raw_ptr_buffer_load_lds' must be a constant integer}}
  __builtin_amdgcn_raw_ptr_buffer_load_lds(rsrc, lds, 4, offset, soffset, 0, x); //expected-error{{argument to '__builtin_amdgcn_raw_ptr_buffer_load_lds' must be a constant integer}}
  __builtin_amdgcn_raw_ptr_buffer_load_lds(rsrc, lds, 3, offset, soffset, 0, 0); //expected-error{{invalid size value}} gfx950-note{{size must be 1, 2, 4, 12 or 16}} gfx90a-note{{size must be 1, 2, or 4}}
}

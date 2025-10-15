// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx908 -S -verify=gfx908,expected -o - %s
// REQUIRES: amdgpu-registered-target

typedef half __attribute__((ext_vector_type(2))) float16x2_t;

void test_raw_ptr_atomics(__amdgpu_buffer_rsrc_t rsrc, float f32, float16x2_t v2f16, int offset, int soffset) {
  f32 = __builtin_amdgcn_raw_ptr_buffer_atomic_fadd_f32(f32, rsrc, offset, soffset, 0); // expected-error{{'__builtin_amdgcn_raw_ptr_buffer_atomic_fadd_f32' needs target feature atomic-fadd-rtn-insts}}
  v2f16 = __builtin_amdgcn_raw_ptr_buffer_atomic_fadd_v2f16(v2f16, rsrc, offset, soffset, 0); // expected-error{{'__builtin_amdgcn_raw_ptr_buffer_atomic_fadd_v2f16' needs target feature atomic-buffer-global-pk-add-f16-insts}}
}

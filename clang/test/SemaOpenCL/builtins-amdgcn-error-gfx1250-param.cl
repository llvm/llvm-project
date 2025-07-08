// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1250 -verify -S -o - %s

typedef int    v4i   __attribute__((ext_vector_type(4)));
typedef int    v8i   __attribute__((ext_vector_type(8)));

void test_setprio_inc_wg(short a) {
  __builtin_amdgcn_s_setprio_inc_wg(a); // expected-error {{'__builtin_amdgcn_s_setprio_inc_wg' must be a constant integer}}
}

void test_s_monitor_sleep(short a) {
  __builtin_amdgcn_s_monitor_sleep(a); // expected-error {{'__builtin_amdgcn_s_monitor_sleep' must be a constant integer}}
}

void test__builtin_amdgcn_cvt_f16_fp8(int a, int b) {
  __builtin_amdgcn_cvt_f16_fp8(a, b); // expected-error {{'__builtin_amdgcn_cvt_f16_fp8' must be a constant integer}}
}

void test__builtin_amdgcn_cvt_f16_bf8(int a, int b) {
  __builtin_amdgcn_cvt_f16_bf8(a, b); // expected-error {{'__builtin_amdgcn_cvt_f16_bf8' must be a constant integer}}
}

void test_amdgcn_tensor_load_store(v4i sg0, v8i sg1, v4i sg2, v4i sg3, int cpol)
{
  __builtin_amdgcn_tensor_load_to_lds(sg0, sg1, sg2, sg3, cpol); // expected-error {{'__builtin_amdgcn_tensor_load_to_lds' must be a constant integer}}
  __builtin_amdgcn_tensor_load_to_lds_d2(sg0, sg1, cpol); // expected-error {{'__builtin_amdgcn_tensor_load_to_lds_d2' must be a constant integer}}
  __builtin_amdgcn_tensor_store_from_lds(sg0, sg1, sg2, sg3, cpol); // expected-error {{'__builtin_amdgcn_tensor_store_from_lds' must be a constant integer}}
  __builtin_amdgcn_tensor_store_from_lds_d2(sg0, sg1, cpol); // expected-error {{'__builtin_amdgcn_tensor_store_from_lds_d2' must be a constant integer}}
}

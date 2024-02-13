// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1210 -verify -S -emit-llvm -o - %s

typedef int    v2i   __attribute__((ext_vector_type(2)));
typedef int    v4i   __attribute__((ext_vector_type(4)));

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

void test_cvt_sr_f8_f16(global int* out, short sr, int old, int sel)
{
  *out = __builtin_amdgcn_cvt_sr_bf8_f16(1.0, sr, old, sel); // expected-error {{'__builtin_amdgcn_cvt_sr_bf8_f16' must be a constant integer}}
  *out = __builtin_amdgcn_cvt_sr_fp8_f16(1.0, sr, old, sel); // expected-error {{'__builtin_amdgcn_cvt_sr_fp8_f16' must be a constant integer}}
}

void test_amdgcn_load_monitor(global int* b32gaddr, global v2i* b64gaddr, global v4i* b128gaddr, int *b32faddr, v2i* b64faddr, v4i *b128faddr,
                              global int* b32out, global v2i* b64out, global v4i* b128out, int cpol)
{
  *b32out  = __builtin_amdgcn_global_load_monitor_b32(b32gaddr, cpol); // expected-error {{'__builtin_amdgcn_global_load_monitor_b32' must be a constant integer}}
  *b64out  = __builtin_amdgcn_global_load_monitor_b64(b64gaddr, cpol); // expected-error {{'__builtin_amdgcn_global_load_monitor_b64' must be a constant integer}}
  *b128out = __builtin_amdgcn_global_load_monitor_b128(b128gaddr, cpol); // expected-error {{'__builtin_amdgcn_global_load_monitor_b128' must be a constant integer}}
  *b32out  = __builtin_amdgcn_flat_load_monitor_b32(b32faddr, cpol); // expected-error {{'__builtin_amdgcn_flat_load_monitor_b32' must be a constant integer}}
  *b64out  = __builtin_amdgcn_flat_load_monitor_b64(b64faddr, cpol); // expected-error {{'__builtin_amdgcn_flat_load_monitor_b64' must be a constant integer}}
  *b128out = __builtin_amdgcn_flat_load_monitor_b128(b128faddr, cpol); // expected-error {{'__builtin_amdgcn_flat_load_monitor_b128' must be a constant integer}}
}

void test_amdgcn_async_load_store_lds(global char* gaddr8, global int *gaddr32, global v2i* gaddr64, global v4i* gaddr128, local char* laddr8,
                                      local int *laddr32, local v2i* laddr64, local v4i* laddr128, int cpol, int mask)
{
  __builtin_amdgcn_cluster_load_async_to_lds_b8(gaddr8, laddr8, cpol, mask); // expected-error {{'__builtin_amdgcn_cluster_load_async_to_lds_b8' must be a constant integer}}
  __builtin_amdgcn_cluster_load_async_to_lds_b32(gaddr32, laddr32, cpol, mask); // expected-error {{'__builtin_amdgcn_cluster_load_async_to_lds_b32' must be a constant integer}}
  __builtin_amdgcn_cluster_load_async_to_lds_b64(gaddr64, laddr64, cpol, mask); // expected-error {{'__builtin_amdgcn_cluster_load_async_to_lds_b64' must be a constant integer}}
  __builtin_amdgcn_cluster_load_async_to_lds_b128(gaddr128, laddr128, cpol, mask); // expected-error {{'__builtin_amdgcn_cluster_load_async_to_lds_b128' must be a constant integer}}

  __builtin_amdgcn_global_store_async_from_lds_b8(gaddr8, laddr8, cpol); // expected-error {{'__builtin_amdgcn_global_store_async_from_lds_b8' must be a constant integer}}
  __builtin_amdgcn_global_store_async_from_lds_b32(gaddr32, laddr32, cpol); // expected-error {{'__builtin_amdgcn_global_store_async_from_lds_b32' must be a constant integer}}
  __builtin_amdgcn_global_store_async_from_lds_b64(gaddr64, laddr64, cpol); // expected-error {{'__builtin_amdgcn_global_store_async_from_lds_b64' must be a constant integer}}
  __builtin_amdgcn_global_store_async_from_lds_b128(gaddr128, laddr128, cpol); // expected-error {{'__builtin_amdgcn_global_store_async_from_lds_b128' must be a constant integer}}

  __builtin_amdgcn_global_store_async_from_lds_b8(gaddr8, laddr8, cpol); // expected-error {{'__builtin_amdgcn_global_store_async_from_lds_b8' must be a constant integer}}
  __builtin_amdgcn_global_store_async_from_lds_b32(gaddr32, laddr32, cpol); // expected-error {{'__builtin_amdgcn_global_store_async_from_lds_b32' must be a constant integer}}
  __builtin_amdgcn_global_store_async_from_lds_b64(gaddr64, laddr64, cpol); // expected-error {{'__builtin_amdgcn_global_store_async_from_lds_b64' must be a constant integer}}
  __builtin_amdgcn_global_store_async_from_lds_b128(gaddr128, laddr128, cpol); // expected-error {{'__builtin_amdgcn_global_store_async_from_lds_b128' must be a constant integer}}
}

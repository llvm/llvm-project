// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1210 -verify -emit-llvm -o - %s

typedef unsigned int uint;
typedef unsigned char uchar;
typedef int    v2i   __attribute__((ext_vector_type(2)));
typedef int    v4i   __attribute__((ext_vector_type(4)));
typedef unsigned int __attribute__((ext_vector_type(2))) uint2;
typedef unsigned int __attribute__((ext_vector_type(3))) uint3;
typedef unsigned int __attribute__((ext_vector_type(6))) uint6;
typedef __bf16 __attribute__((ext_vector_type(8))) bfloat8;
typedef __bf16 __attribute__((ext_vector_type(32))) bfloat32;
typedef half __attribute__((ext_vector_type(8))) half8;
typedef half __attribute__((ext_vector_type(32))) half32;
typedef float __attribute__((ext_vector_type(8))) float8;
typedef float __attribute__((ext_vector_type(32))) float32;


void test_setprio_inc_wg(short a) {
  __builtin_amdgcn_s_setprio_inc_wg(a); // expected-error {{'__builtin_amdgcn_s_setprio_inc_wg' must be a constant integer}}
}

void test_s_monitor_sleep(short a) {
  __builtin_amdgcn_s_monitor_sleep(a); // expected-error {{'__builtin_amdgcn_s_monitor_sleep' must be a constant integer}}
}

void test_s_wait_asynccnt(short a) {
  __builtin_amdgcn_s_wait_asynccnt(a); // expected-error {{'__builtin_amdgcn_s_wait_asynccnt' must be a constant integer}}
}

void test_s_wait_tensorcnt(short a) {
  __builtin_amdgcn_s_wait_tensorcnt(a); // expected-error {{'__builtin_amdgcn_s_wait_tensorcnt' must be a constant integer}}
}

void test__builtin_amdgcn_cvt_f16_fp8(int a, int b) {
  __builtin_amdgcn_cvt_f16_fp8(a, b); // expected-error {{'__builtin_amdgcn_cvt_f16_fp8' must be a constant integer}}
}

void test__builtin_amdgcn_cvt_f16_bf8(int a, int b) {
  __builtin_amdgcn_cvt_f16_bf8(a, b); // expected-error {{'__builtin_amdgcn_cvt_f16_bf8' must be a constant integer}}
}

void test_cvt_sr_f8_f16(global int* out, uint sr, int old, int sel)
{
  *out = __builtin_amdgcn_cvt_sr_bf8_f16(1.0, sr, old, sel); // expected-error {{'__builtin_amdgcn_cvt_sr_bf8_f16' must be a constant integer}}
  *out = __builtin_amdgcn_cvt_sr_fp8_f16(1.0, sr, old, sel); // expected-error {{'__builtin_amdgcn_cvt_sr_fp8_f16' must be a constant integer}}
}

void test_cvt_scale_pk(global half32 *outh32, global bfloat32 *outy32, uint6 src6,
                       global half8 *outh8, global bfloat8 *outy8, uint2 src2,
                       global float32 *outf32, global float8 *outf8, uint src1,
                       uint scale, uchar scale_sel)
{
  *outh32 = __builtin_amdgcn_cvt_scale_pk32_f16_fp6(src6, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk32_f16_fp6' must be a constant integer}}
  *outy32 = __builtin_amdgcn_cvt_scale_pk32_bf16_fp6(src6, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk32_bf16_fp6' must be a constant integer}}
  *outh32 = __builtin_amdgcn_cvt_scale_pk32_f16_bf6(src6, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk32_f16_bf6' must be a constant integer}}
  *outy32 = __builtin_amdgcn_cvt_scale_pk32_bf16_bf6(src6, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk32_bf16_bf6' must be a constant integer}}
  *outh8 = __builtin_amdgcn_cvt_scale_pk8_f16_fp8(src2, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk8_f16_fp8' must be a constant integer}}
  *outy8 = __builtin_amdgcn_cvt_scale_pk8_bf16_fp8(src2, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk8_bf16_fp8' must be a constant integer}}
  *outh8 = __builtin_amdgcn_cvt_scale_pk8_f16_bf8(src2, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk8_f16_bf8' must be a constant integer}}
  *outy8 = __builtin_amdgcn_cvt_scale_pk8_bf16_bf8(src2, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk8_bf16_bf8' must be a constant integer}}
  *outh8 = __builtin_amdgcn_cvt_scale_pk8_f16_fp4(src1, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk8_f16_fp4' must be a constant integer}}
  *outy8 = __builtin_amdgcn_cvt_scale_pk8_bf16_fp4(src1, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk8_bf16_fp4' must be a constant integer}}
  *outf32 = __builtin_amdgcn_cvt_scale_pk32_f32_fp6(src6, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk32_f32_fp6' must be a constant integer}}
  *outf32 = __builtin_amdgcn_cvt_scale_pk32_f32_bf6(src6, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk32_f32_bf6' must be a constant integer}}
  *outf8 = __builtin_amdgcn_cvt_scale_pk8_f32_fp8(src2, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk8_f32_fp8' must be a constant integer}}
  *outf8 = __builtin_amdgcn_cvt_scale_pk8_f32_bf8(src2, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk8_f32_bf8' must be a constant integer}}
  *outf8 = __builtin_amdgcn_cvt_scale_pk8_f32_fp4(src1, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk8_f32_fp4' must be a constant integer}}
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

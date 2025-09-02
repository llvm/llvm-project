// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-- -target-cpu gfx1250 -verify -S -o - %s

typedef unsigned int uint;
typedef unsigned short int ushort;
typedef int    v2i   __attribute__((ext_vector_type(2)));
typedef unsigned int __attribute__((ext_vector_type(2))) uint2;
typedef unsigned int __attribute__((ext_vector_type(3))) uint3;
typedef __bf16 __attribute__((ext_vector_type(8))) bfloat8;
typedef __bf16 __attribute__((ext_vector_type(16))) bfloat16;
typedef __bf16 __attribute__((ext_vector_type(32))) bfloat32;
typedef half __attribute__((ext_vector_type(8))) half8;
typedef half __attribute__((ext_vector_type(16))) half16;
typedef half __attribute__((ext_vector_type(32))) half32;
typedef float __attribute__((ext_vector_type(8))) float8;
typedef float __attribute__((ext_vector_type(16))) float16;
typedef float __attribute__((ext_vector_type(32))) float32;

typedef int    v4i   __attribute__((ext_vector_type(4)));
typedef int    v8i   __attribute__((ext_vector_type(8)));

void test_setprio_inc_wg(short a) {
  __builtin_amdgcn_s_setprio_inc_wg(a); // expected-error {{'__builtin_amdgcn_s_setprio_inc_wg' must be a constant integer}}
}

void test_bitop3_args(global uint* out, uint a, uint b, uint c) {
  *out = __builtin_amdgcn_bitop3_b32(a, b, c, a); // expected-error {{argument to '__builtin_amdgcn_bitop3_b32' must be a constant integer}}
  *out = __builtin_amdgcn_bitop3_b16((ushort)a, (ushort)b, (ushort)c, a); // expected-error {{argument to '__builtin_amdgcn_bitop3_b16' must be a constant integer}}
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

void test_cvt_scale_pk(global half8 *outh8, global bfloat8 *outy8, uint2 src2,
                       global float32 *outf32, global half16 *outh16, global bfloat16 *outy16,
                       global float16 *outf16, uint3 src3,
                       global float8 *outf8, uint src1, uint scale, uint scale_sel)
{
  *outh8 = __builtin_amdgcn_cvt_scale_pk8_f16_fp8(src2, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk8_f16_fp8' must be a constant integer}}
  *outy8 = __builtin_amdgcn_cvt_scale_pk8_bf16_fp8(src2, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk8_bf16_fp8' must be a constant integer}}
  *outh8 = __builtin_amdgcn_cvt_scale_pk8_f16_bf8(src2, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk8_f16_bf8' must be a constant integer}}
  *outy8 = __builtin_amdgcn_cvt_scale_pk8_bf16_bf8(src2, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk8_bf16_bf8' must be a constant integer}}
  *outh8 = __builtin_amdgcn_cvt_scale_pk8_f16_fp4(src1, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk8_f16_fp4' must be a constant integer}}
  *outy8 = __builtin_amdgcn_cvt_scale_pk8_bf16_fp4(src1, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk8_bf16_fp4' must be a constant integer}}
  *outf8 = __builtin_amdgcn_cvt_scale_pk8_f32_fp8(src2, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk8_f32_fp8' must be a constant integer}}
  *outf8 = __builtin_amdgcn_cvt_scale_pk8_f32_bf8(src2, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk8_f32_bf8' must be a constant integer}}
  *outf8 = __builtin_amdgcn_cvt_scale_pk8_f32_fp4(src1, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk8_f32_fp4' must be a constant integer}}
  *outh16 = __builtin_amdgcn_cvt_scale_pk16_f16_fp6(src3, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk16_f16_fp6' must be a constant integer}}
  *outy16 = __builtin_amdgcn_cvt_scale_pk16_bf16_fp6(src3, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk16_bf16_fp6' must be a constant integer}}
  *outh16 = __builtin_amdgcn_cvt_scale_pk16_f16_bf6(src3, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk16_f16_bf6' must be a constant integer}}
  *outy16 = __builtin_amdgcn_cvt_scale_pk16_bf16_bf6(src3, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk16_bf16_bf6' must be a constant integer}}
  *outf16 = __builtin_amdgcn_cvt_scale_pk16_f32_fp6(src3, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk16_f32_fp6' must be a constant integer}}
  *outf16 = __builtin_amdgcn_cvt_scale_pk16_f32_bf6(src3, scale, scale_sel); // expected-error {{'__builtin_amdgcn_cvt_scale_pk16_f32_bf6' must be a constant integer}}

  *outh8 = __builtin_amdgcn_cvt_scale_pk8_f16_fp8(src2, scale, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  *outy8 = __builtin_amdgcn_cvt_scale_pk8_bf16_fp8(src2, scale, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  *outh8 = __builtin_amdgcn_cvt_scale_pk8_f16_bf8(src2, scale, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  *outy8 = __builtin_amdgcn_cvt_scale_pk8_bf16_bf8(src2, scale, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  *outh8 = __builtin_amdgcn_cvt_scale_pk8_f16_fp4(src1, scale, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  *outy8 = __builtin_amdgcn_cvt_scale_pk8_bf16_fp4(src1, scale, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  *outf8 = __builtin_amdgcn_cvt_scale_pk8_f32_fp8(src2, scale, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  *outf8 = __builtin_amdgcn_cvt_scale_pk8_f32_bf8(src2, scale, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  *outf8 = __builtin_amdgcn_cvt_scale_pk8_f32_fp4(src1, scale, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  *outh16 = __builtin_amdgcn_cvt_scale_pk16_f16_fp6(src3, scale, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  *outy16 = __builtin_amdgcn_cvt_scale_pk16_bf16_fp6(src3, scale, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  *outh16 = __builtin_amdgcn_cvt_scale_pk16_f16_bf6(src3, scale, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  *outy16 = __builtin_amdgcn_cvt_scale_pk16_bf16_bf6(src3, scale, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  *outf16 = __builtin_amdgcn_cvt_scale_pk16_f32_fp6(src3, scale, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  *outf16 = __builtin_amdgcn_cvt_scale_pk16_f32_bf6(src3, scale, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
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

void test_amdgcn_cluster_load(global int* addr32, global v2i* addr64, global v4i* addr128, global int* b32out, global v2i* b64out, global v4i* b128out, int cpol, int mask)
{
  *b32out  = __builtin_amdgcn_cluster_load_b32(addr32, cpol, mask); // expected-error {{'__builtin_amdgcn_cluster_load_b32' must be a constant integer}}
  *b64out  = __builtin_amdgcn_cluster_load_b64(addr64, cpol, mask); // expected-error {{'__builtin_amdgcn_cluster_load_b64' must be a constant integer}}
  *b128out = __builtin_amdgcn_cluster_load_b128(addr128, cpol, mask); // expected-error {{'__builtin_amdgcn_cluster_load_b128' must be a constant integer}}
}

void test_amdgcn_async_load_store_lds_offset(global char* gaddr8, global int *gaddr32, global v2i* gaddr64, global v4i* gaddr128, local char* laddr8,
                                             local int *laddr32, local v2i* laddr64, local v4i* laddr128, int offset, int mask)
{
  __builtin_amdgcn_global_store_async_from_lds_b8(gaddr8, laddr8, offset, 0); // expected-error {{'__builtin_amdgcn_global_store_async_from_lds_b8' must be a constant integer}}
  __builtin_amdgcn_global_store_async_from_lds_b32(gaddr32, laddr32, offset, 0); // expected-error {{'__builtin_amdgcn_global_store_async_from_lds_b32' must be a constant integer}}
  __builtin_amdgcn_global_store_async_from_lds_b64(gaddr64, laddr64, offset, 0); // expected-error {{'__builtin_amdgcn_global_store_async_from_lds_b64' must be a constant integer}}
  __builtin_amdgcn_global_store_async_from_lds_b128(gaddr128, laddr128, offset, 0); // expected-error {{'__builtin_amdgcn_global_store_async_from_lds_b128' must be a constant integer}}

  __builtin_amdgcn_global_store_async_from_lds_b8(gaddr8, laddr8, offset, 0); // expected-error {{'__builtin_amdgcn_global_store_async_from_lds_b8' must be a constant integer}}
  __builtin_amdgcn_global_store_async_from_lds_b32(gaddr32, laddr32, offset, 0); // expected-error {{'__builtin_amdgcn_global_store_async_from_lds_b32' must be a constant integer}}
  __builtin_amdgcn_global_store_async_from_lds_b64(gaddr64, laddr64, offset, 0); // expected-error {{'__builtin_amdgcn_global_store_async_from_lds_b64' must be a constant integer}}
  __builtin_amdgcn_global_store_async_from_lds_b128(gaddr128, laddr128, offset, 0); // expected-error {{'__builtin_amdgcn_global_store_async_from_lds_b128' must be a constant integer}}
}

void test_amdgcn_async_load_store_lds_cpol(global char* gaddr8, global int *gaddr32, global v2i* gaddr64, global v4i* gaddr128, local char* laddr8,
                                           local int *laddr32, local v2i* laddr64, local v4i* laddr128, int cpol, int mask)
{
  __builtin_amdgcn_global_store_async_from_lds_b8(gaddr8, laddr8, 16, cpol); // expected-error {{'__builtin_amdgcn_global_store_async_from_lds_b8' must be a constant integer}}
  __builtin_amdgcn_global_store_async_from_lds_b32(gaddr32, laddr32, 16, cpol); // expected-error {{'__builtin_amdgcn_global_store_async_from_lds_b32' must be a constant integer}}
  __builtin_amdgcn_global_store_async_from_lds_b64(gaddr64, laddr64, 16, cpol); // expected-error {{'__builtin_amdgcn_global_store_async_from_lds_b64' must be a constant integer}}
  __builtin_amdgcn_global_store_async_from_lds_b128(gaddr128, laddr128, 16, cpol); // expected-error {{'__builtin_amdgcn_global_store_async_from_lds_b128' must be a constant integer}}

  __builtin_amdgcn_global_store_async_from_lds_b8(gaddr8, laddr8, 16, cpol); // expected-error {{'__builtin_amdgcn_global_store_async_from_lds_b8' must be a constant integer}}
  __builtin_amdgcn_global_store_async_from_lds_b32(gaddr32, laddr32, 16, cpol); // expected-error {{'__builtin_amdgcn_global_store_async_from_lds_b32' must be a constant integer}}
  __builtin_amdgcn_global_store_async_from_lds_b64(gaddr64, laddr64, 16, cpol); // expected-error {{'__builtin_amdgcn_global_store_async_from_lds_b64' must be a constant integer}}
  __builtin_amdgcn_global_store_async_from_lds_b128(gaddr128, laddr128, 16, cpol); // expected-error {{'__builtin_amdgcn_global_store_async_from_lds_b128' must be a constant integer}}
}

void test_amdgcn_tensor_load_store(v4i sg0, v8i sg1, v4i sg2, v4i sg3, int cpol)
{
  __builtin_amdgcn_tensor_load_to_lds(sg0, sg1, sg2, sg3, cpol); // expected-error {{'__builtin_amdgcn_tensor_load_to_lds' must be a constant integer}}
  __builtin_amdgcn_tensor_load_to_lds_d2(sg0, sg1, cpol); // expected-error {{'__builtin_amdgcn_tensor_load_to_lds_d2' must be a constant integer}}
  __builtin_amdgcn_tensor_store_from_lds(sg0, sg1, sg2, sg3, cpol); // expected-error {{'__builtin_amdgcn_tensor_store_from_lds' must be a constant integer}}
  __builtin_amdgcn_tensor_store_from_lds_d2(sg0, sg1, cpol); // expected-error {{'__builtin_amdgcn_tensor_store_from_lds_d2' must be a constant integer}}
}

void test_prefetch(generic void *fptr, global void *gptr, int cpol) {
  __builtin_amdgcn_flat_prefetch(fptr, cpol); // expected-error {{'__builtin_amdgcn_flat_prefetch' must be a constant integer}}
  __builtin_amdgcn_global_prefetch(gptr, cpol); // expected-error {{'__builtin_amdgcn_global_prefetch' must be a constant integer}}
}

void test_cvt_f32_fp8_e5m3(global int* out, int a)
{
  *out = __builtin_amdgcn_cvt_f32_fp8_e5m3(a, a); // expected-error {{'__builtin_amdgcn_cvt_f32_fp8_e5m3' must be a constant integer}}
}

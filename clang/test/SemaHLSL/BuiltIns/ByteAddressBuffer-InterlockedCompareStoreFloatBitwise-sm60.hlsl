// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header \
// RUN:   -triple dxil-pc-shadermodel6.0-library %s -fsyntax-only -verify \
// RUN:   -verify-ignore-unexpected=warning

// The float-bitwise compare reuses the 32-bit integer DXIL operation, so it
// needs no capability bits and works from SM 6.0. The 64-bit compare-store
// needs SM 6.6. This file checks both halves, so it proves the two are gated
// differently.

RWByteAddressBuffer BAB : register(u0);
RasterizerOrderedByteAddressBuffer ROVB : register(u1);
groupshared float gs_f32;
groupshared uint64_t gs_u64;

void sm60_bab_float_bitwise_ok(uint off, float cmp, float v) {
  BAB.InterlockedCompareStoreFloatBitwise(off, cmp, v);
}

void sm60_rovb_float_bitwise_ok(uint off, float cmp, float v) {
  ROVB.InterlockedCompareStoreFloatBitwise(off, cmp, v);
}

void sm60_free_function_ok(float cmp, float v) {
  InterlockedCompareStoreFloatBitwise(gs_f32, cmp, v);
}

void sm60_direct_builtin_ok(float cmp, float v) {
  __builtin_hlsl_interlocked_compare_store_float_bitwise(gs_f32, cmp, v);
}

void sm60_no_bab_compare_store64(uint off, uint64_t cmp, uint64_t v) {
  BAB.InterlockedCompareStore64(off, cmp, v);
  // expected-error@-1 {{no member named 'InterlockedCompareStore64' in 'hlsl::RWByteAddressBuffer'}}
}

void sm60_no_direct_builtin_u64(uint64_t cmp, uint64_t v) {
  __builtin_hlsl_interlocked_compare_store(gs_u64, cmp, v);
  // expected-error@-1 {{'__builtin_hlsl_interlocked_compare_store' requires shader model 6.6 or newer}}
}

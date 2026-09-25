// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header \
// RUN:   -triple dxil-pc-shadermodel6.0-library %s -fsyntax-only -verify \
// RUN:   -verify-ignore-unexpected=warning

// The float exchange reuses the 32-bit integer DXIL operation, so it needs no
// capability bits and works from SM 6.0. The 64-bit exchange needs SM 6.6.
// This file checks both halves, so it proves the two are gated differently.

RWByteAddressBuffer BAB : register(u0);
RasterizerOrderedByteAddressBuffer ROVB : register(u1);
groupshared float gs_f32;
groupshared int64_t gs_i64;

void sm60_bab_float_ok(uint off, float v, out float orig) {
  BAB.InterlockedExchangeFloat(off, v, orig);
}

void sm60_rovb_float_ok(uint off, float v, out float orig) {
  ROVB.InterlockedExchangeFloat(off, v, orig);
}

void sm60_free_function_ok(float v) {
  float orig;
  InterlockedExchange(gs_f32, v, orig);
}

void sm60_direct_builtin_ok(float v) {
  float orig;
  __builtin_hlsl_interlocked_exchange(gs_f32, v, orig);
}

void sm60_no_bab_exchange64(uint off, uint64_t v, out uint64_t orig) {
  BAB.InterlockedExchange64(off, v, orig);
  // expected-error@-1 {{no member named 'InterlockedExchange64' in 'hlsl::RWByteAddressBuffer'}}
}

void sm60_no_direct_builtin_i64(int64_t v, out int64_t orig) {
  __builtin_hlsl_interlocked_exchange(gs_i64, v, orig);
  // expected-error@-1 {{'__builtin_hlsl_interlocked_exchange' requires shader model 6.6 or newer}}
}

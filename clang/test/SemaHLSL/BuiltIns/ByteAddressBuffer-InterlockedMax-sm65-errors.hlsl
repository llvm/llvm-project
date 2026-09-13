// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header \
// RUN:   -triple dxil-pc-shadermodel6.5-library %s -fsyntax-only -verify \
// RUN:   -verify-ignore-unexpected=warning

RWByteAddressBuffer BAB : register(u0);
RasterizerOrderedByteAddressBuffer ROVB : register(u1);

void sm65_no_bab_max64(uint off, int64_t v) {
  BAB.InterlockedMax64(off, v);
  // expected-error@-1 {{no member named 'InterlockedMax64' in 'hlsl::RWByteAddressBuffer'}}
}

void sm65_no_rovb_max64(uint off, int64_t v) {
  ROVB.InterlockedMax64(off, v);
  // expected-error@-1 {{no member named 'InterlockedMax64' in 'hlsl::RasterizerOrderedByteAddressBuffer'}}
}

void sm65_bab_max32_ok(uint off, int v) {
  BAB.InterlockedMax(off, v);
}

groupshared int64_t gs_i64;
void sm65_direct_builtin(int64_t v) {
  __builtin_hlsl_interlocked_max(gs_i64, v);
  // expected-error@-1 {{'__builtin_hlsl_interlocked_max' requires shader model 6.6 or newer}}
}

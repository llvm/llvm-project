// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header \
// RUN:   -triple dxil-pc-shadermodel6.5-library %s -fsyntax-only -verify \
// RUN:   -verify-ignore-unexpected=warning

RWByteAddressBuffer BAB : register(u0);
RasterizerOrderedByteAddressBuffer ROVB : register(u1);

void sm65_no_bab_min64(uint off, int64_t v) {
  BAB.InterlockedMin64(off, v);
  // expected-error@-1 {{no member named 'InterlockedMin64' in 'hlsl::RWByteAddressBuffer'}}
}

void sm65_no_rovb_min64(uint off, int64_t v) {
  ROVB.InterlockedMin64(off, v);
  // expected-error@-1 {{no member named 'InterlockedMin64' in 'hlsl::RasterizerOrderedByteAddressBuffer'}}
}

void sm65_bab_min32_ok(uint off, int v) {
  BAB.InterlockedMin(off, v);
}

groupshared int64_t gs_i64;
void sm65_direct_builtin(int64_t v) {
  __builtin_hlsl_interlocked_min(gs_i64, v);
  // expected-error@-1 {{'__builtin_hlsl_interlocked_min' requires shader model 6.6 or newer}}
}

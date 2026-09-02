// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header \
// RUN:   -triple dxil-pc-shadermodel6.5-library %s -fsyntax-only -verify \
// RUN:   -verify-ignore-unexpected=warning

RWByteAddressBuffer BAB : register(u0);
RasterizerOrderedByteAddressBuffer ROVB : register(u1);

// InterlockedAdd64 is only synthesized on DXIL when the shader model is at
// least 6.6 (matches DXC). On SM 6.5 the member is not declared at all, so
// the reference must fail with "no member named".
void sm65_no_bab_add64(uint off, int64_t v) {
  BAB.InterlockedAdd64(off, v);
  // expected-error@-1 {{no member named 'InterlockedAdd64' in 'hlsl::RWByteAddressBuffer'}}
}

void sm65_no_rovb_add64(uint off, int64_t v) {
  ROVB.InterlockedAdd64(off, v);
  // expected-error@-1 {{no member named 'InterlockedAdd64' in 'hlsl::RasterizerOrderedByteAddressBuffer'}}
}

// 32-bit InterlockedAdd is always available.
void sm65_bab_add32_ok(uint off, int v) {
  BAB.InterlockedAdd(off, v);
}

// Direct calls to the 64-bit interlocked builtin must also be rejected with a
// clear source-location error on pre-SM6.6 DXIL targets.
groupshared int64_t gs_i64;
void sm65_direct_builtin(int64_t v) {
  __builtin_hlsl_interlocked_add(gs_i64, v);
  // expected-error@-1 {{'__builtin_hlsl_interlocked_add' requires shader model 6.6 or newer}}
}

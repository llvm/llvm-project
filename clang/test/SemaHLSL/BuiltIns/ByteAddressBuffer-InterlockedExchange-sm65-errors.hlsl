// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header \
// RUN:   -triple dxil-pc-shadermodel6.5-library %s -fsyntax-only -verify \
// RUN:   -verify-ignore-unexpected=warning

RWByteAddressBuffer BAB : register(u0);
RasterizerOrderedByteAddressBuffer ROVB : register(u1);

void sm65_no_bab_exchange64(uint off, uint64_t v, out uint64_t orig) {
  BAB.InterlockedExchange64(off, v, orig);
  // expected-error@-1 {{no member named 'InterlockedExchange64' in 'hlsl::RWByteAddressBuffer'}}
}

void sm65_no_rovb_exchange64(uint off, uint64_t v, out uint64_t orig) {
  ROVB.InterlockedExchange64(off, v, orig);
  // expected-error@-1 {{no member named 'InterlockedExchange64' in 'hlsl::RasterizerOrderedByteAddressBuffer'}}
}

void sm65_bab_exchange32_ok(uint off, uint v, out uint orig) {
  BAB.InterlockedExchange(off, v, orig);
}

groupshared int64_t gs_i64;
void sm65_direct_builtin(int64_t v, out int64_t orig) {
  __builtin_hlsl_interlocked_exchange(gs_i64, v, orig);
  // expected-error@-1 {{'__builtin_hlsl_interlocked_exchange' requires shader model 6.6 or newer}}
}

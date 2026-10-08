// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header \
// RUN:   -triple dxil-pc-shadermodel6.5-library %s -fsyntax-only -verify \
// RUN:   -verify-ignore-unexpected=warning

RWByteAddressBuffer BAB : register(u0);
RasterizerOrderedByteAddressBuffer ROVB : register(u1);

void sm65_no_bab_compare_store64(uint off, uint64_t cmp, uint64_t v) {
  BAB.InterlockedCompareStore64(off, cmp, v);
  // expected-error@-1 {{no member named 'InterlockedCompareStore64' in 'hlsl::RWByteAddressBuffer'}}
}

void sm65_no_rovb_compare_store64(uint off, uint64_t cmp, uint64_t v) {
  ROVB.InterlockedCompareStore64(off, cmp, v);
  // expected-error@-1 {{no member named 'InterlockedCompareStore64' in 'hlsl::RasterizerOrderedByteAddressBuffer'}}
}

void sm65_bab_compare_store32_ok(uint off, uint cmp, uint v) {
  BAB.InterlockedCompareStore(off, cmp, v);
}

groupshared int64_t gs_i64;
void sm65_direct_builtin(int64_t cmp, int64_t v) {
  __builtin_hlsl_interlocked_compare_store(gs_i64, cmp, v);
  // expected-error@-1 {{'__builtin_hlsl_interlocked_compare_store' requires shader model 6.6 or newer}}
}

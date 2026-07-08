// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header \
// RUN:   -triple dxil-pc-shadermodel6.6-compute %s -fsyntax-only -verify \
// RUN:   -verify-ignore-unexpected=note,warning

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header \
// RUN:   -triple dxil-pc-shadermodel6.5-compute -DTEST_SM65 %s -fsyntax-only \
// RUN:   -verify -verify-ignore-unexpected=note,warning

RWByteAddressBuffer BAB : register(u0);
RasterizerOrderedByteAddressBuffer ROVB : register(u1);

struct S { int x; };

#ifdef TEST_SM65

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

#else

void too_few(uint off) {
  BAB.InterlockedAdd(off);
  // expected-error@-1 {{no matching member function for call to 'InterlockedAdd'}}
}

void too_many(uint off, int v, int extra) {
  int orig;
  BAB.InterlockedAdd(off, v, orig, extra);
  // expected-error@-1 {{no matching member function for call to 'InterlockedAdd'}}
}

void struct_value(uint off, S v) {
  BAB.InterlockedAdd(off, v);
  // expected-error@-1 {{no matching member function for call to 'InterlockedAdd'}}
}

// Same shape of errors on RasterizerOrderedByteAddressBuffer.
void rovb_too_few(uint off) {
  ROVB.InterlockedAdd(off);
  // expected-error@-1 {{no matching member function for call to 'InterlockedAdd'}}
}

void rovb_struct_value(uint off, S v) {
  ROVB.InterlockedAdd(off, v);
  // expected-error@-1 {{no matching member function for call to 'InterlockedAdd'}}
}

#endif

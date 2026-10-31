// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header \
// RUN:   -triple dxil-pc-shadermodel6.6-library %s -fsyntax-only -verify \
// RUN:   -verify-ignore-unexpected=note,warning

RWByteAddressBuffer BAB : register(u0);
RasterizerOrderedByteAddressBuffer ROVB : register(u1);

struct S { int x; };

void too_few(uint off) {
  BAB.InterlockedMin(off);
  // expected-error@-1 {{no matching member function for call to 'InterlockedMin'}}
}

void too_many(uint off, int v, int extra) {
  int orig;
  BAB.InterlockedMin(off, v, orig, extra);
  // expected-error@-1 {{no matching member function for call to 'InterlockedMin'}}
}

void struct_value(uint off, S v) {
  BAB.InterlockedMin(off, v);
  // expected-error@-1 {{no matching member function for call to 'InterlockedMin'}}
}

void rovb_too_few(uint off) {
  ROVB.InterlockedMin(off);
  // expected-error@-1 {{no matching member function for call to 'InterlockedMin'}}
}

void rovb_struct_value(uint off, S v) {
  ROVB.InterlockedMin(off, v);
  // expected-error@-1 {{no matching member function for call to 'InterlockedMin'}}
}

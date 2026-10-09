// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header \
// RUN:   -triple dxil-pc-shadermodel6.6-library %s -fsyntax-only -verify \
// RUN:   -verify-ignore-unexpected=note,warning

RWByteAddressBuffer BAB : register(u0);
RasterizerOrderedByteAddressBuffer ROVB : register(u1);

struct S { int x; };

void too_few(uint off) {
  BAB.InterlockedMax(off);
  // expected-error@-1 {{no matching member function for call to 'InterlockedMax'}}
}

void too_many(uint off, int v, int extra) {
  int orig;
  BAB.InterlockedMax(off, v, orig, extra);
  // expected-error@-1 {{no matching member function for call to 'InterlockedMax'}}
}

void struct_value(uint off, S v) {
  BAB.InterlockedMax(off, v);
  // expected-error@-1 {{no matching member function for call to 'InterlockedMax'}}
}

void rovb_too_few(uint off) {
  ROVB.InterlockedMax(off);
  // expected-error@-1 {{no matching member function for call to 'InterlockedMax'}}
}

void rovb_struct_value(uint off, S v) {
  ROVB.InterlockedMax(off, v);
  // expected-error@-1 {{no matching member function for call to 'InterlockedMax'}}
}

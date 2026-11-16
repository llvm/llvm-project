// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header \
// RUN:   -triple dxil-pc-shadermodel6.6-library %s -fsyntax-only -verify \
// RUN:   -verify-ignore-unexpected=note,warning

// notes that are ignored are strictly ones of the form:
// (frontend): candidate function not viable: requires X arguments, but Y was provided
// or
// (frontend): candidate function not viable: no known conversion from X to Y for Nth argument
// which is in line with expectations, but is difficult to exactly match since
// the notes are not tied to explicit source lines, but just (frontend).

RWByteAddressBuffer BAB : register(u0);
RasterizerOrderedByteAddressBuffer ROVB : register(u1);

struct S { int x; };

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

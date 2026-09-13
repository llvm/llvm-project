// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header \
// RUN:   -triple dxil-pc-shadermodel6.6-library %s -fsyntax-only -verify \
// RUN:   -verify-ignore-unexpected=note,warning

// Unlike the other interlocked methods, InterlockedExchange declares a single
// overload per element type because the original value is required. There is
// therefore no overload set to fail against, so Clang reports the argument
// mismatch directly instead of 'no matching member function'.

RWByteAddressBuffer BAB : register(u0);
RasterizerOrderedByteAddressBuffer ROVB : register(u1);

struct S { int x; };

void too_few(uint off) {
  BAB.InterlockedExchange(off);
  // expected-error@-1 {{too few arguments to function call, expected 3, have 1}}
}

void missing_original_value(uint off, uint v) {
  BAB.InterlockedExchange(off, v);
  // expected-error@-1 {{too few arguments to function call, expected 3, have 2}}
}

void too_many(uint off, uint v, uint extra) {
  uint orig;
  BAB.InterlockedExchange(off, v, orig, extra);
  // expected-error@-1 {{too many arguments to function call, expected 3, have 4}}
}

void struct_value(uint off, S v) {
  uint orig;
  BAB.InterlockedExchange(off, v, orig);
  // expected-error@-1 {{cannot initialize a parameter of type 'unsigned int' with an lvalue of type 'S'}}
}

void rovb_missing_original_value(uint off, uint v) {
  ROVB.InterlockedExchange(off, v);
  // expected-error@-1 {{too few arguments to function call, expected 3, have 2}}
}

void rovb_struct_value(uint off, S v) {
  uint orig;
  ROVB.InterlockedExchange(off, v, orig);
  // expected-error@-1 {{cannot initialize a parameter of type 'unsigned int' with an lvalue of type 'S'}}
}

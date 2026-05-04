// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header \
// RUN:   -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only \
// RUN:   -disable-llvm-passes -verify

void too_few(int v) {
  int dest;
  InterlockedAdd(dest); // expected-error{{no matching function for call to 'InterlockedAdd'}}
  // expected-note@hlsl/hlsl_alias_intrinsics.h:* 8 {{candidate function}}
}

void too_many(int v, int extra) {
  int dest;
  int o;
  InterlockedAdd(dest, v, o, extra); // expected-error{{no matching function for call to 'InterlockedAdd'}}
  // expected-note@hlsl/hlsl_alias_intrinsics.h:* 8 {{candidate function}}
}

void float_dest(float v) {
  float dest;
  InterlockedAdd(dest, v); // expected-error{{call to 'InterlockedAdd' is ambiguous}}
  // expected-note@hlsl/hlsl_alias_intrinsics.h:* 4 {{candidate function}}
}

void bool_dest(bool v) {
  bool dest;
  InterlockedAdd(dest, v); // expected-error{{call to 'InterlockedAdd' is ambiguous}}
  // expected-note@hlsl/hlsl_alias_intrinsics.h:* 3 {{candidate function}}
}

struct S { int x; };

void struct_dest(int v) {
  S s;
  InterlockedAdd(s, v); // expected-error{{no matching function for call to 'InterlockedAdd'}}
  // expected-note@hlsl/hlsl_alias_intrinsics.h:* 8 {{candidate function}}
}

void mismatched_type(int v) {
  int dest;
  uint orig;
  InterlockedAdd(dest, v, orig); // expected-error{{call to 'InterlockedAdd' is ambiguous}}
  // expected-note@hlsl/hlsl_alias_intrinsics.h:* 4 {{candidate function}}
}

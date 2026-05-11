// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header \
// RUN:   -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only \
// RUN:   -disable-llvm-passes -verify

// InterlockedAdd is provided as a set of address-space-qualified overloads
// (groupshared/device, {int,uint,int64_t,uint64_t}, 2-arg/3-arg). All arg
// mismatches surface as "no matching function" with 16 candidates. The
// candidate notes come from synthesized FunctionDecls with no source
// location, so they are matched with `@*:*`.

groupshared int   gs_i32;
groupshared float gs_f32;
struct S { int x; };
groupshared S     gs_s;

void too_few(int v) {
  InterlockedAdd(gs_i32); // expected-error{{no matching function for call to 'InterlockedAdd'}}
  // expected-note@*:* 16 {{candidate function}}
}

void too_many(int v, int extra) {
  int o;
  InterlockedAdd(gs_i32, v, o, extra); // expected-error{{no matching function for call to 'InterlockedAdd'}}
  // expected-note@*:* 16 {{candidate function}}
}

// Atomics must operate on actual addresses in groupshared or device memory;
// passing a plain local (no address space) must not bind to any overload.
void local_dest(int v) {
  int dest;
  InterlockedAdd(dest, v); // expected-error{{no matching function for call to 'InterlockedAdd'}}
  // expected-note@*:* 16 {{candidate function}}
}

void float_dest(float v) {
  InterlockedAdd(gs_f32, v); // expected-error{{no matching function for call to 'InterlockedAdd'}}
  // expected-note@*:* 16 {{candidate function}}
}

void struct_dest(int v) {
  InterlockedAdd(gs_s, v); // expected-error{{no matching function for call to 'InterlockedAdd'}}
  // expected-note@*:* 16 {{candidate function}}
}

void mismatched_orig_type(int v) {
  uint orig;
  InterlockedAdd(gs_i32, v, orig); // expected-error{{no matching function for call to 'InterlockedAdd'}}
  // expected-note@*:* 16 {{candidate function}}
}

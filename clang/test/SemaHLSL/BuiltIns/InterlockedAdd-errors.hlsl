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

// The tests below exercise direct invocations of the underlying clang builtin
// `__builtin_hlsl_interlocked_add`. These bypass overload resolution against
// the synthesized `InterlockedAdd` overload set (the builtin's prototype in
// Builtins.td is `void (...)`), so each error is produced by the explicit
// checks in SemaHLSL.cpp rather than by candidate-set rejection.

void direct_too_few() {
  __builtin_hlsl_interlocked_add(gs_i32);
  // expected-error@-1 {{too few arguments to function call, expected at least 2, have 1}}
}

void direct_too_many(int v, int extra) {
  int o;
  __builtin_hlsl_interlocked_add(gs_i32, v, o, extra);
  // expected-error@-1 {{too many arguments to function call, expected at most 3, have 4}}
}

void direct_non_integer_dest() {
  S local_s;
  __builtin_hlsl_interlocked_add(local_s, 1);
  // expected-error@-1 {{1st argument must be a scalar integer type (was 'S')}}
}

void direct_nonlvalue_dest(int v) {
  __builtin_hlsl_interlocked_add(1, v);
  // expected-error@-1 {{cannot bind non-lvalue argument '1' to out parameter}}
}

void direct_mismatched_value() {
  uint uv = 1u;
  __builtin_hlsl_interlocked_add(gs_i32, uv);
  // expected-error@-1 {{passing 'uint' (aka 'unsigned int') to parameter of incompatible type 'int'}}
}

void direct_mismatched_orig(int v) {
  uint orig;
  __builtin_hlsl_interlocked_add(gs_i32, v, orig);
  // expected-error@-1 {{passing 'uint' (aka 'unsigned int') to parameter of incompatible type 'int'}}
}

void direct_nonlvalue_orig(int v) {
  __builtin_hlsl_interlocked_add(gs_i32, v, 1);
  // expected-error@-1 {{cannot bind non-lvalue argument '1' to out parameter}}
}

void direct_default_as_dest(int v) {
  int local;
  __builtin_hlsl_interlocked_add(local, v);
  // expected-error@-1 {{1st argument to atomic builtin must reference groupshared or device memory (was 'int')}}
}

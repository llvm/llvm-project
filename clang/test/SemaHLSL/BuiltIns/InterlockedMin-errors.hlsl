// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header \
// RUN:   -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only \
// RUN:   -disable-llvm-passes -verify

// InterlockedMin is provided as a set of address-space-qualified overloads
// (groupshared/device, {int,uint,int64_t,uint64_t}, 2-arg/3-arg).

groupshared int gs_i32;
groupshared float gs_f32;
struct S { int x; };
groupshared S gs_s;

void too_few() {
  InterlockedMin(gs_i32); // expected-error{{no matching function for call to 'InterlockedMin'}}
  // expected-note@*:* 16 {{candidate function}}
}

void too_many(int v, int extra) {
  int orig;
  InterlockedMin(gs_i32, v, orig, extra); // expected-error{{no matching function for call to 'InterlockedMin'}}
  // expected-note@*:* 16 {{candidate function}}
}

void local_dest(int v) {
  int dest;
  InterlockedMin(dest, v); // expected-error{{no matching function for call to 'InterlockedMin'}}
  // expected-note@*:* 16 {{candidate function}}
}

void float_dest(float v) {
  InterlockedMin(gs_f32, v); // expected-error{{no matching function for call to 'InterlockedMin'}}
  // expected-note@*:* 16 {{candidate function}}
}

void struct_dest(int v) {
  InterlockedMin(gs_s, v); // expected-error{{no matching function for call to 'InterlockedMin'}}
  // expected-note@*:* 16 {{candidate function}}
}

void mismatched_orig_type(int v) {
  uint orig;
  InterlockedMin(gs_i32, v, orig); // expected-error{{no matching function for call to 'InterlockedMin'}}
  // expected-note@*:* 16 {{candidate function}}
}

void direct_too_few() {
  __builtin_hlsl_interlocked_min(gs_i32);
  // expected-error@-1 {{too few arguments to function call, expected at least 2, have 1}}
}

void direct_too_many(int v, int extra) {
  int orig;
  __builtin_hlsl_interlocked_min(gs_i32, v, orig, extra);
  // expected-error@-1 {{too many arguments to function call, expected at most 3, have 4}}
}

void direct_non_integer_dest() {
  S local_s;
  __builtin_hlsl_interlocked_min(local_s, 1);
  // expected-error@-1 {{1st argument must be a scalar integer type (was 'S')}}
}

void direct_nonlvalue_dest(int v) {
  __builtin_hlsl_interlocked_min(1, v);
  // expected-error@-1 {{cannot bind non-lvalue argument '1' to out parameter}}
}

void direct_mismatched_value() {
  uint value = 1;
  __builtin_hlsl_interlocked_min(gs_i32, value);
  // expected-error@-1 {{passing 'uint' (aka 'unsigned int') to parameter of incompatible type 'int'}}
}

void direct_mismatched_orig(int v) {
  uint orig;
  __builtin_hlsl_interlocked_min(gs_i32, v, orig);
  // expected-error@-1 {{passing 'uint' (aka 'unsigned int') to parameter of incompatible type 'int'}}
}

void direct_nonlvalue_orig(int v) {
  __builtin_hlsl_interlocked_min(gs_i32, v, 1);
  // expected-error@-1 {{cannot bind non-lvalue argument '1' to out parameter}}
}

void direct_default_as_dest(int v) {
  int local;
  __builtin_hlsl_interlocked_min(local, v);
  // expected-error@-1 {{1st argument to atomic builtin must reference groupshared or device memory (was 'int')}}
}

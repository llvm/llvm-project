// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header \
// RUN:   -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only \
// RUN:   -disable-llvm-passes -verify

// InterlockedExchange is provided as a set of address-space-qualified
// overloads (groupshared/device, {int,uint,int64_t,uint64_t}). It always
// reports the previous value, so there is no 2-argument form.

groupshared int gs_i32;
groupshared float gs_f32;
struct S { int x; };
groupshared S gs_s;

void too_few() {
  InterlockedExchange(gs_i32); // expected-error{{no matching function for call to 'InterlockedExchange'}}
  // expected-note@*:* 8 {{candidate function}}
}

void missing_original_value(int v) {
  InterlockedExchange(gs_i32, v); // expected-error{{no matching function for call to 'InterlockedExchange'}}
  // expected-note@*:* 8 {{candidate function}}
}

void too_many(int v, int extra) {
  int orig;
  InterlockedExchange(gs_i32, v, orig, extra); // expected-error{{no matching function for call to 'InterlockedExchange'}}
  // expected-note@*:* 8 {{candidate function}}
}

void local_dest(int v) {
  int dest;
  int orig;
  InterlockedExchange(dest, v, orig); // expected-error{{no matching function for call to 'InterlockedExchange'}}
  // expected-note@*:* 8 {{candidate function}}
}

void float_dest(float v) {
  float orig;
  InterlockedExchange(gs_f32, v, orig); // expected-error{{no matching function for call to 'InterlockedExchange'}}
  // expected-note@*:* 8 {{candidate function}}
}

void struct_dest(int v) {
  int orig;
  InterlockedExchange(gs_s, v, orig); // expected-error{{no matching function for call to 'InterlockedExchange'}}
  // expected-note@*:* 8 {{candidate function}}
}

void mismatched_orig_type(int v) {
  uint orig;
  InterlockedExchange(gs_i32, v, orig); // expected-error{{no matching function for call to 'InterlockedExchange'}}
  // expected-note@*:* 8 {{candidate function}}
}

void direct_too_few() {
  __builtin_hlsl_interlocked_exchange(gs_i32);
  // expected-error@-1 {{too few arguments to function call, expected 3, have 1}}
}

void direct_missing_original_value(int v) {
  __builtin_hlsl_interlocked_exchange(gs_i32, v);
  // expected-error@-1 {{too few arguments to function call, expected 3, have 2}}
}

void direct_too_many(int v, int extra) {
  int orig;
  __builtin_hlsl_interlocked_exchange(gs_i32, v, orig, extra);
  // expected-error@-1 {{too many arguments to function call, expected 3, have 4}}
}

void direct_non_integer_dest() {
  S local_s;
  S orig;
  __builtin_hlsl_interlocked_exchange(local_s, 1, orig);
  // expected-error@-1 {{1st argument must be a scalar integer type (was 'S')}}
}

void direct_nonlvalue_dest(int v) {
  int orig;
  __builtin_hlsl_interlocked_exchange(1, v, orig);
  // expected-error@-1 {{cannot bind non-lvalue argument '1' to out parameter}}
}

void direct_mismatched_value() {
  uint value = 1;
  int orig;
  __builtin_hlsl_interlocked_exchange(gs_i32, value, orig);
  // expected-error@-1 {{passing 'uint' (aka 'unsigned int') to parameter of incompatible type 'int'}}
}

void direct_mismatched_orig(int v) {
  uint orig;
  __builtin_hlsl_interlocked_exchange(gs_i32, v, orig);
  // expected-error@-1 {{passing 'uint' (aka 'unsigned int') to parameter of incompatible type 'int'}}
}

void direct_nonlvalue_orig(int v) {
  __builtin_hlsl_interlocked_exchange(gs_i32, v, 1);
  // expected-error@-1 {{cannot bind non-lvalue argument '1' to out parameter}}
}

void direct_default_as_dest(int v) {
  int local;
  int orig;
  __builtin_hlsl_interlocked_exchange(local, v, orig);
  // expected-error@-1 {{1st argument to atomic builtin must reference groupshared or device memory (was 'int')}}
}

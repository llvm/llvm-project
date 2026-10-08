// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header \
// RUN:   -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only \
// RUN:   -disable-llvm-passes -verify

// InterlockedCompareExchange is provided as a set of address-space-qualified
// overloads (groupshared/device, {int,uint,int64_t,uint64_t}). It reports the
// value that was in the destination, so it has a single 4-argument form with a
// trailing out parameter.

groupshared int gs_i32;
groupshared float gs_f32;
struct S { int x; };
groupshared S gs_s;

void too_few(int cmp, int v) {
  InterlockedCompareExchange(gs_i32, cmp, v); // expected-error{{no matching function for call to 'InterlockedCompareExchange'}}
  // expected-note@*:* 8 {{candidate function}}
}

void too_many(int cmp, int v, int extra) {
  int orig;
  InterlockedCompareExchange(gs_i32, cmp, v, orig, extra); // expected-error{{no matching function for call to 'InterlockedCompareExchange'}}
  // expected-note@*:* 8 {{candidate function}}
}

void local_dest(int cmp, int v) {
  int dest;
  int orig;
  InterlockedCompareExchange(dest, cmp, v, orig); // expected-error{{no matching function for call to 'InterlockedCompareExchange'}}
  // expected-note@*:* 8 {{candidate function}}
}

void float_dest(float cmp, float v) {
  float orig;
  InterlockedCompareExchange(gs_f32, cmp, v, orig); // expected-error{{no matching function for call to 'InterlockedCompareExchange'}}
  // expected-note@*:* 8 {{candidate function}}
}

void struct_dest(int cmp, int v) {
  S orig;
  InterlockedCompareExchange(gs_s, cmp, v, orig); // expected-error{{no matching function for call to 'InterlockedCompareExchange'}}
  // expected-note@*:* 8 {{candidate function}}
}

// The out parameter is a reference, so it cannot bind across types.
void mismatched_orig_type(int cmp, int v) {
  float orig;
  InterlockedCompareExchange(gs_i32, cmp, v, orig); // expected-error{{no matching function for call to 'InterlockedCompareExchange'}}
  // expected-note@*:* 8 {{candidate function}}
}

void direct_too_few(int cmp, int v) {
  __builtin_hlsl_interlocked_compare_exchange(gs_i32, cmp, v);
  // expected-error@-1 {{too few arguments to function call, expected 4, have 3}}
}

void direct_too_many(int cmp, int v, int extra) {
  int orig;
  __builtin_hlsl_interlocked_compare_exchange(gs_i32, cmp, v, orig, extra);
  // expected-error@-1 {{too many arguments to function call, expected 4, have 5}}
}

void direct_non_integer_dest() {
  S local_s;
  int orig;
  __builtin_hlsl_interlocked_compare_exchange(local_s, 1, 2, orig);
  // expected-error@-1 {{1st argument must be a scalar integer type (was 'S')}}
}

void direct_float_dest(float cmp, float v) {
  float orig;
  __builtin_hlsl_interlocked_compare_exchange(gs_f32, cmp, v, orig);
  // expected-error@-1 {{1st argument must be a scalar integer type (was 'float')}}
}

void direct_nonlvalue_dest(int cmp, int v) {
  int orig;
  __builtin_hlsl_interlocked_compare_exchange(1, cmp, v, orig);
  // expected-error@-1 {{cannot bind non-lvalue argument '1' to out parameter}}
}

// The last argument is an out parameter, so an rvalue is rejected there.
void direct_nonlvalue_original_value(int cmp, int v) {
  __builtin_hlsl_interlocked_compare_exchange(gs_i32, cmp, v, 0);
  // expected-error@-1 {{cannot bind non-lvalue argument '0' to out parameter}}
}

void direct_mismatched_compare() {
  uint cmp = 1;
  int orig;
  __builtin_hlsl_interlocked_compare_exchange(gs_i32, cmp, 2, orig);
  // expected-error@-1 {{passing 'uint' (aka 'unsigned int') to parameter of incompatible type 'int'}}
}

void direct_mismatched_value() {
  uint v = 1;
  int orig;
  __builtin_hlsl_interlocked_compare_exchange(gs_i32, 1, v, orig);
  // expected-error@-1 {{passing 'uint' (aka 'unsigned int') to parameter of incompatible type 'int'}}
}

void direct_mismatched_original_value() {
  uint orig;
  __builtin_hlsl_interlocked_compare_exchange(gs_i32, 1, 2, orig);
  // expected-error@-1 {{passing 'uint' (aka 'unsigned int') to parameter of incompatible type 'int'}}
}

void direct_default_as_dest(int cmp, int v) {
  int local;
  int orig;
  __builtin_hlsl_interlocked_compare_exchange(local, cmp, v, orig);
  // expected-error@-1 {{1st argument to atomic builtin must reference groupshared or device memory (was 'int')}}
}

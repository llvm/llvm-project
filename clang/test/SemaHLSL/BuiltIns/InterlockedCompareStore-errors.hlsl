// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header \
// RUN:   -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only \
// RUN:   -disable-llvm-passes -verify

// InterlockedCompareStore is provided as a set of address-space-qualified
// overloads (groupshared/device, {int,uint,int64_t,uint64_t}). It reports
// nothing, so it has a single 3-argument form and no out parameter.

groupshared int gs_i32;
groupshared float gs_f32;
struct S { int x; };
groupshared S gs_s;

void too_few(int cmp) {
  InterlockedCompareStore(gs_i32, cmp); // expected-error{{no matching function for call to 'InterlockedCompareStore'}}
  // expected-note@*:* 8 {{candidate function}}
}

void too_many(int cmp, int v, int extra) {
  InterlockedCompareStore(gs_i32, cmp, v, extra); // expected-error{{no matching function for call to 'InterlockedCompareStore'}}
  // expected-note@*:* 8 {{candidate function}}
}

void local_dest(int cmp, int v) {
  int dest;
  InterlockedCompareStore(dest, cmp, v); // expected-error{{no matching function for call to 'InterlockedCompareStore'}}
  // expected-note@*:* 8 {{candidate function}}
}

void float_dest(float cmp, float v) {
  InterlockedCompareStore(gs_f32, cmp, v); // expected-error{{no matching function for call to 'InterlockedCompareStore'}}
  // expected-note@*:* 8 {{candidate function}}
}

void struct_dest(int cmp, int v) {
  InterlockedCompareStore(gs_s, cmp, v); // expected-error{{no matching function for call to 'InterlockedCompareStore'}}
  // expected-note@*:* 8 {{candidate function}}
}

void direct_too_few(int cmp) {
  __builtin_hlsl_interlocked_compare_store(gs_i32, cmp);
  // expected-error@-1 {{too few arguments to function call, expected 3, have 2}}
}

void direct_too_many(int cmp, int v, int extra) {
  __builtin_hlsl_interlocked_compare_store(gs_i32, cmp, v, extra);
  // expected-error@-1 {{too many arguments to function call, expected 3, have 4}}
}

void direct_non_integer_dest() {
  S local_s;
  __builtin_hlsl_interlocked_compare_store(local_s, 1, 2);
  // expected-error@-1 {{1st argument must be a scalar integer type (was 'S')}}
}

void direct_float_dest(float cmp, float v) {
  __builtin_hlsl_interlocked_compare_store(gs_f32, cmp, v);
  // expected-error@-1 {{1st argument must be a scalar integer type (was 'float')}}
}

void direct_nonlvalue_dest(int cmp, int v) {
  __builtin_hlsl_interlocked_compare_store(1, cmp, v);
  // expected-error@-1 {{cannot bind non-lvalue argument '1' to out parameter}}
}

void direct_mismatched_compare() {
  uint cmp = 1;
  __builtin_hlsl_interlocked_compare_store(gs_i32, cmp, 2);
  // expected-error@-1 {{passing 'uint' (aka 'unsigned int') to parameter of incompatible type 'int'}}
}

void direct_mismatched_value() {
  uint v = 1;
  __builtin_hlsl_interlocked_compare_store(gs_i32, 1, v);
  // expected-error@-1 {{passing 'uint' (aka 'unsigned int') to parameter of incompatible type 'int'}}
}

void direct_default_as_dest(int cmp, int v) {
  int local;
  __builtin_hlsl_interlocked_compare_store(local, cmp, v);
  // expected-error@-1 {{1st argument to atomic builtin must reference groupshared or device memory (was 'int')}}
}

// Unlike the read-modify-write operations, the third argument is the new value
// rather than an out parameter, so an rvalue is accepted here.
void direct_rvalue_value_ok() {
  __builtin_hlsl_interlocked_compare_store(gs_i32, 1, 2);
}

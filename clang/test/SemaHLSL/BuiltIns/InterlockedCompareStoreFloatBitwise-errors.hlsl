// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header \
// RUN:   -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only \
// RUN:   -disable-llvm-passes -verify

// InterlockedCompareStoreFloatBitwise compares the bit pattern of a 32-bit
// float, so it is provided as a float-only overload set (groupshared/device).
// It reports nothing, so it has a single 3-argument form and no out parameter.

groupshared float gs_f32;
groupshared int gs_i32;
groupshared double gs_f64;
groupshared half gs_f16;
struct S { float x; };
groupshared S gs_s;

void too_few(float cmp) {
  InterlockedCompareStoreFloatBitwise(gs_f32, cmp); // expected-error{{no matching function for call to 'InterlockedCompareStoreFloatBitwise'}}
  // expected-note@*:* 2 {{candidate function}}
}

void too_many(float cmp, float v, float extra) {
  InterlockedCompareStoreFloatBitwise(gs_f32, cmp, v, extra); // expected-error{{no matching function for call to 'InterlockedCompareStoreFloatBitwise'}}
  // expected-note@*:* 2 {{candidate function}}
}

void local_dest(float cmp, float v) {
  float dest;
  InterlockedCompareStoreFloatBitwise(dest, cmp, v); // expected-error{{no matching function for call to 'InterlockedCompareStoreFloatBitwise'}}
  // expected-note@*:* 2 {{candidate function}}
}

// The bitwise compare is defined for 32-bit float alone, so there is no
// integer, half or double overload.
void int_dest(int cmp, int v) {
  InterlockedCompareStoreFloatBitwise(gs_i32, cmp, v); // expected-error{{no matching function for call to 'InterlockedCompareStoreFloatBitwise'}}
  // expected-note@*:* 2 {{candidate function}}
}

void double_dest(double cmp, double v) {
  InterlockedCompareStoreFloatBitwise(gs_f64, cmp, v); // expected-error{{no matching function for call to 'InterlockedCompareStoreFloatBitwise'}}
  // expected-note@*:* 2 {{candidate function}}
}

void half_dest(half cmp, half v) {
  InterlockedCompareStoreFloatBitwise(gs_f16, cmp, v); // expected-error{{no matching function for call to 'InterlockedCompareStoreFloatBitwise'}}
  // expected-note@*:* 2 {{candidate function}}
}

void struct_dest(float cmp, float v) {
  InterlockedCompareStoreFloatBitwise(gs_s, cmp, v); // expected-error{{no matching function for call to 'InterlockedCompareStoreFloatBitwise'}}
  // expected-note@*:* 2 {{candidate function}}
}

void direct_too_few(float cmp) {
  __builtin_hlsl_interlocked_compare_store_float_bitwise(gs_f32, cmp);
  // expected-error@-1 {{too few arguments to function call, expected 3, have 2}}
}

void direct_too_many(float cmp, float v, float extra) {
  __builtin_hlsl_interlocked_compare_store_float_bitwise(gs_f32, cmp, v, extra);
  // expected-error@-1 {{too many arguments to function call, expected 3, have 4}}
}

void direct_integer_dest(int cmp, int v) {
  __builtin_hlsl_interlocked_compare_store_float_bitwise(gs_i32, cmp, v);
  // expected-error@-1 {{1st argument must be a scalar 32 bit floating-point type (was 'int')}}
}

void direct_double_dest(double cmp, double v) {
  __builtin_hlsl_interlocked_compare_store_float_bitwise(gs_f64, cmp, v);
  // expected-error@-1 {{1st argument must be a scalar 32 bit floating-point type (was 'double')}}
}

void direct_half_dest(half cmp, half v) {
  __builtin_hlsl_interlocked_compare_store_float_bitwise(gs_f16, cmp, v);
  // expected-error@-1 {{1st argument must be a scalar 32 bit floating-point type (was 'half')}}
}

void direct_non_scalar_dest() {
  S local_s;
  __builtin_hlsl_interlocked_compare_store_float_bitwise(local_s, 1.0f, 2.0f);
  // expected-error@-1 {{1st argument must be a scalar 32 bit floating-point type (was 'S')}}
}

void direct_nonlvalue_dest(float cmp, float v) {
  __builtin_hlsl_interlocked_compare_store_float_bitwise(1.0f, cmp, v);
  // expected-error@-1 {{cannot bind non-lvalue argument '1.F' to out parameter}}
}

void direct_default_as_dest(float cmp, float v) {
  float local;
  __builtin_hlsl_interlocked_compare_store_float_bitwise(local, cmp, v);
  // expected-error@-1 {{1st argument to atomic builtin must reference groupshared or device memory (was 'float')}}
}

// The last argument is the new value rather than an out parameter, so an
// rvalue is accepted here.
void direct_rvalue_value_ok() {
  __builtin_hlsl_interlocked_compare_store_float_bitwise(gs_f32, 1.0f, 2.0f);
}

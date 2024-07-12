// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -o - -fsyntax-only %s -verify
// XFAIL: *

// https://github.com/llvm/llvm-project/issues/81047

// expected-no-diagnostics
void Fn4(int64_t2 L);
void Fn4(int2 I);

void Call4(int16_t H) { Fn4(H); }

int test_builtin_dot_bool_type_promotion(bool p0, bool p1) {
  return dot(p0, p1);
}

float test_dot_scalar_mismatch(float p0, int p1) { return dot(p0, p1); }

float test_dot_element_type_mismatch(int2 p0, float2 p1) { return dot(p0, p1); }

float test_builtin_dot_vec_int_to_float_promotion(int2 p0, float2 p1) {
  return dot(p0, p1);
}

int64_t test_builtin_dot_vec_int_to_int64_promotion(int64_t2 p0, int2 p1) {
  return dot(p0, p1);
}

float test_builtin_dot_vec_half_to_float_promotion(float2 p0, half2 p1) {
  return dot(p0, p1);
}

float test_builtin_dot_vec_int16_to_float_promotion(float2 p0, int16_t2 p1) {
  return dot(p0, p1);
}

half test_builtin_dot_vec_int16_to_half_promotion(half2 p0, int16_t2 p1) {
  return dot(p0, p1);
}

int test_builtin_dot_vec_int16_to_int_promotion(int2 p0, int16_t2 p1) {
  return dot(p0, p1);
}

int64_t test_builtin_dot_vec_int16_to_int64_promotion(int64_t2 p0,
                                                      int16_t2 p1) {
  return dot(p0, p1);
}

float4 test_frac_int4(int4 p0) { return frac(p0); }

float test_frac_int(int p0) { return frac(p0); }

float test_frac_bool(bool p0) { return frac(p0); }

// This resolves the wrong overload. In clang this converts down to an int, in
// DXC it extends the scalar to a vector.
void Fn(int) {}
void Fn(vector<int64_t,2>) {}

void Call() {
  int64_t V;
  Fn(V);
}

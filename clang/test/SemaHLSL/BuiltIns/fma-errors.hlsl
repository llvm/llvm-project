// RUN: %clang_cc1 -finclude-default-header -fnative-half-type -x hlsl \
// RUN:   -triple dxil-pc-shadermodel6.6-library %s \
// RUN:   -emit-llvm-only -disable-llvm-passes -verify \
// RUN:   -verify-ignore-unexpected=note
// RUN: %clang_cc1 -finclude-default-header -fnative-half-type -x hlsl \
// RUN:   -triple spirv-unknown-vulkan-compute %s \
// RUN:   -emit-llvm-only -disable-llvm-passes -verify \
// RUN:   -verify-ignore-unexpected=note

float bad_float(float a, float b, float c) {
  return fma(a, b, c);
  // expected-error@-1 {{1st argument must be a scalar, vector, or matrix of double type (was 'float')}}
}

float2 bad_float2(float2 a, float2 b, float2 c) {
  return fma(a, b, c);
  // expected-error@-1 {{1st argument must be a scalar, vector, or matrix of double type (was 'float2' (aka 'vector<float, 2>'))}}
}

float2x2 bad_float2x2(float2x2 a, float2x2 b, float2x2 c) {
  return fma(a, b, c);
  // expected-error@-1 {{1st argument must be a scalar, vector, or matrix of double type (was 'float2x2' (aka 'matrix<float, 2, 2>'))}}
}

half bad_half(half a, half b, half c) {
  return fma(a, b, c);
  // expected-error@-1 {{1st argument must be a scalar, vector, or matrix of double type (was 'half')}}
}

half2 bad_half2(half2 a, half2 b, half2 c) {
  return fma(a, b, c);
  // expected-error@-1 {{1st argument must be a scalar, vector, or matrix of double type (was 'half2' (aka 'vector<half, 2>'))}}
}

half2x2 bad_half2x2(half2x2 a, half2x2 b, half2x2 c) {
  return fma(a, b, c);
  // expected-error@-1 {{1st argument must be a scalar, vector, or matrix of double type (was 'half2x2' (aka 'matrix<half, 2, 2>'))}}
}

double mixed_bad_second(double a, float b, double c) {
  return fma(a, b, c);
  // expected-error@-1 {{arguments are of different types ('double' vs 'float')}}
}

double mixed_bad_third(double a, double b, half c) {
  return fma(a, b, c);
  // expected-error@-1 {{arguments are of different types ('double' vs 'half')}}
}

double2 mixed_bad_second_vec(double2 a, float2 b, double2 c) {
  return fma(a, b, c);
  // expected-error@-1 {{arguments are of different types ('vector<double, [...]>' vs 'vector<float, [...]>')}}
}

double2 mixed_bad_third_vec(double2 a, double2 b, float2 c) {
  return fma(a, b, c);
  // expected-error@-1 {{arguments are of different types ('vector<double, [...]>' vs 'vector<float, [...]>')}}
}

double2x2 mixed_bad_second_mat(double2x2 a, float2x2 b, double2x2 c) {
  return fma(a, b, c);
  // expected-error@-1 {{arguments are of different types ('matrix<double, [2 * ...]>' vs 'matrix<float, [2 * ...]>')}}
}

double2x2 mixed_bad_third_mat(double2x2 a, double2x2 b, half2x2 c) {
  return fma(a, b, c);
  // expected-error@-1 {{arguments are of different types ('matrix<double, [2 * ...]>' vs 'matrix<half, [2 * ...]>')}}
}

double shape_mismatch_second(double a, double2 b, double c) {
  return fma(a, b, c);
  // expected-error@-1 {{call to 'fma' is ambiguous}}
}

double2 shape_mismatch_third(double2 a, double2 b, double c) {
  return fma(a, b, c);
  // expected-error@-1 {{call to 'fma' is ambiguous}}
}

double2x2 shape_mismatch_scalar_mat(double2x2 a, double b, double2x2 c) {
  return fma(a, b, c);
  // expected-error@-1 {{call to 'fma' is ambiguous}}
}

double2x2 shape_mismatch_vec_mat(double2x2 a, double2 b, double2x2 c) {
  return fma(a, b, c);
  // expected-error@-1 {{arguments are of different types ('double2x2' (aka 'matrix<double, 2, 2>') vs 'double2' (aka 'vector<double, 2>'))}}
}

int bad_int(int a, int b, int c) {
  return fma(a, b, c);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}
}

int2 bad_int2(int2 a, int2 b, int2 c) {
  return fma(a, b, c);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int2' (aka 'vector<int, 2>'))}}
}

bool bad_bool(bool a, bool b, bool c) {
  return fma(a, b, c);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'bool')}}
}

bool2 bad_bool2(bool2 a, bool2 b, bool2 c) {
  return fma(a, b, c);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'bool2' (aka 'vector<bool, 2>'))}}
}

bool2x2 bad_bool2x2(bool2x2 a, bool2x2 b, bool2x2 c) {
  return fma(a, b, c);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'bool2x2' (aka 'matrix<bool, 2, 2>'))}}
}

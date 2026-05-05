// RUN: %clang_cc1 -finclude-default-header -fnative-half-type -x hlsl \
// RUN:   -triple dxil-pc-shadermodel6.6-library %s -Wdouble-promotion \
// RUN:   -emit-llvm-only -disable-llvm-passes -verify \
// RUN:   -verify-ignore-unexpected=note
// RUN: %clang_cc1 -finclude-default-header -fnative-half-type -x hlsl \
// RUN:   -triple spirv-unknown-vulkan-compute %s -Wdouble-promotion \
// RUN:   -emit-llvm-only -disable-llvm-passes -verify \
// RUN:   -verify-ignore-unexpected=note

float bad_float(float a, float b, float c) {
  // Overload resolution selects 'double fma(double, double, double)'; args promoted to double.
  return fma(a, b, c);
  // expected-warning@-1 3{{implicit conversion increases floating-point precision: 'float' to 'double'}}
  // expected-warning@-2 {{implicit conversion loses floating-point precision: 'double' to 'float'}}
}

float2 bad_float2(float2 a, float2 b, float2 c) {
  // Overload resolution selects 'double2 fma(double2, double2, double2)'; args promoted to double2.
  return fma(a, b, c);
  // expected-warning@-1 3{{implicit conversion increases floating-point precision: 'float2' (aka 'vector<float, 2>') to 'vector<double, 2>' (vector of 2 'double' values)}}
  // expected-warning@-2 {{implicit conversion loses floating-point precision: 'vector<double, 2>' (vector of 2 'double' values) to 'vector<float, 2>' (vector of 2 'float' values)}}
}

float2x2 bad_float2x2(float2x2 a, float2x2 b, float2x2 c) {
  // Overload resolution selects 'double2x2 fma(double2x2, double2x2, double2x2)'; args promoted.
  return fma(a, b, c);
  // expected-warning@-1 3{{implicit conversion increases floating-point precision: 'float2x2' (aka 'matrix<float, 2, 2>') to 'matrix<double, 2, 2>'}}
  // expected-warning@-2 {{implicit conversion loses floating-point precision: 'matrix<double, 2, 2>' to 'matrix<float, 2, 2>'}}
}

half bad_half(half a, half b, half c) {
  // Overload resolution selects 'double fma(double, double, double)'; args promoted to double.
  return fma(a, b, c);
  // expected-warning@-1 3{{implicit conversion increases floating-point precision: 'half' to 'double'}}
  // expected-warning@-2 {{implicit conversion loses floating-point precision: 'double' to 'half'}}
}

half2 bad_half2(half2 a, half2 b, half2 c) {
  // Overload resolution selects 'double2 fma(double2, double2, double2)'; args promoted.
  return fma(a, b, c);
  // expected-warning@-1 3{{implicit conversion increases floating-point precision: 'half2' (aka 'vector<half, 2>') to 'vector<double, 2>' (vector of 2 'double' values)}}
  // expected-warning@-2 {{implicit conversion loses floating-point precision: 'vector<double, 2>' (vector of 2 'double' values) to 'vector<half, 2>' (vector of 2 'half' values)}}
}

half2x2 bad_half2x2(half2x2 a, half2x2 b, half2x2 c) {
  // Overload resolution selects 'double2x2 fma(double2x2, double2x2, double2x2)'; args promoted.
  return fma(a, b, c);
  // expected-warning@-1 3{{implicit conversion increases floating-point precision: 'half2x2' (aka 'matrix<half, 2, 2>') to 'matrix<double, 2, 2>'}}
  // expected-warning@-2 {{implicit conversion loses floating-point precision: 'matrix<double, 2, 2>' to 'matrix<half, 2, 2>'}}
}

double mixed_bad_second(double a, float b, double c) {
  // Overload resolution selects 'double fma(double, double, double)'; float promoted to double.
  return fma(a, b, c);
  // expected-warning@-1 {{implicit conversion increases floating-point precision: 'float' to 'double'}}
}

double mixed_bad_third(double a, double b, half c) {
  // Overload resolution selects 'double fma(double, double, double)'; half promoted to double.
  return fma(a, b, c);
  // expected-warning@-1 {{implicit conversion increases floating-point precision: 'half' to 'double'}}
}

double2 mixed_bad_second_vec(double2 a, float2 b, double2 c) {
  // Overload resolution selects 'double2 fma(double2, double2, double2)'; float2 promoted.
  return fma(a, b, c);
  // expected-warning@-1 {{implicit conversion increases floating-point precision: 'float2' (aka 'vector<float, 2>') to 'vector<double, 2>' (vector of 2 'double' values)}}
}

double2 mixed_bad_third_vec(double2 a, double2 b, float2 c) {
  // Overload resolution selects 'double2 fma(double2, double2, double2)'; float2 promoted.
  return fma(a, b, c);
  // expected-warning@-1 {{implicit conversion increases floating-point precision: 'float2' (aka 'vector<float, 2>') to 'vector<double, 2>' (vector of 2 'double' values)}}
}

double2x2 mixed_bad_second_mat(double2x2 a, float2x2 b, double2x2 c) {
  // Overload resolution selects 'double2x2 fma(double2x2, double2x2, double2x2)'; float2x2 promoted.
  return fma(a, b, c);
  // expected-warning@-1 {{implicit conversion increases floating-point precision: 'float2x2' (aka 'matrix<float, 2, 2>') to 'matrix<double, 2, 2>'}}
}

double2x2 mixed_bad_third_mat(double2x2 a, double2x2 b, half2x2 c) {
  // Overload resolution selects 'double2x2 fma(double2x2, double2x2, double2x2)'; half2x2 promoted.
  return fma(a, b, c);
  // expected-warning@-1 {{implicit conversion increases floating-point precision: 'half2x2' (aka 'matrix<half, 2, 2>') to 'matrix<double, 2, 2>'}}
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
  // Overload resolution selects the scalar 'double fma(double, double, double)'
  // overload; vector/matrix arguments are truncated to scalar.
  // expected-warning@-3 {{implicit conversion turns matrix to scalar: 'double2x2' (aka 'matrix<double, 2, 2>') to 'double'}}
  // expected-warning@-4 {{implicit conversion turns vector to scalar: 'double2' (aka 'vector<double, 2>') to 'double'}}
  // expected-warning@-5 {{implicit conversion turns matrix to scalar: 'double2x2' (aka 'matrix<double, 2, 2>') to 'double'}}
}

int bad_int(int a, int b, int c) {
  // Overload resolution selects 'double fma(double, double, double)'; ints promoted to double.
  return fma(a, b, c);
  // expected-warning@-1 {{implicit conversion turns floating-point number into integer: 'double' to 'int'}}
}

int2 bad_int2(int2 a, int2 b, int2 c) {
  // Overload resolution selects 'double2 fma(double2, double2, double2)'; int2 promoted.
  return fma(a, b, c);
  // expected-warning@-1 {{implicit conversion turns floating-point number into integer: 'vector<double, 2>' (vector of 2 'double' values) to 'vector<int, 2>' (vector of 2 'int' values)}}
}

bool bad_bool(bool a, bool b, bool c) {
  // Overload resolution selects 'double fma(double, double, double)'; bools promoted to double.
  return fma(a, b, c);
  // expected-warning@-1 {{implicit conversion turns floating-point number into integer: 'double' to 'bool'}}
  // expected-warning@-2 {{implicit conversion turns floating-point number into bool: 'double' to 'bool'}}
}

bool2 bad_bool2(bool2 a, bool2 b, bool2 c) {
  // Overload resolution selects 'double2 fma(double2, double2, double2)'; bool2 promoted.
  return fma(a, b, c);
  // expected-warning@-1 {{implicit conversion turns floating-point number into integer: 'vector<double, 2>' (vector of 2 'double' values) to 'vector<bool, 2>' (vector of 2 'bool' values)}}
}

bool2x2 bad_bool2x2(bool2x2 a, bool2x2 b, bool2x2 c) {
  // Overload resolution selects 'double2x2 fma(double2x2, double2x2, double2x2)'; bool2x2 promoted.
  return fma(a, b, c);
  // expected-warning@-1 {{implicit conversion turns floating-point number into integer: 'matrix<double, 2, 2>' to 'matrix<bool, 2, 2>'}}
}

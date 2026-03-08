// RUN: %clang_cc1 -finclude-default-header -fnative-half-type -x hlsl \
// RUN:   -triple dxil-pc-shadermodel6.6-library %s -DTEST_DXIL \
// RUN:   -emit-llvm-only -disable-llvm-passes -verify=dxil
// RUN: %clang_cc1 -finclude-default-header -fnative-half-type -x hlsl \
// RUN:   -triple spirv-unknown-vulkan-compute %s -DTEST_SPIRV \
// RUN:   -emit-llvm-only -disable-llvm-passes -verify=spv

#ifdef TEST_DXIL
float dxil_fma_float(float a, float b, float c) {
  return fma(a, b, c);
  // dxil-error@-1 {{1st argument must be a scalar, vector, or matrix of double type (was 'float')}}
}

float2 dxil_fma_float2(float2 a, float2 b, float2 c) {
  return fma(a, b, c);
  // dxil-error@-1 {{1st argument must be a scalar, vector, or matrix of double type (was 'float2' (aka 'vector<float, 2>'))}}
}

float4 dxil_fma_float4(float4 a, float4 b, float4 c) {
  return fma(a, b, c);
  // dxil-error@-1 {{1st argument must be a scalar, vector, or matrix of double type (was 'float4' (aka 'vector<float, 4>'))}}
}

float2x2 dxil_fma_float2x2(float2x2 a, float2x2 b, float2x2 c) {
  return fma(a, b, c);
  // dxil-error@-1 {{1st argument must be a scalar, vector, or matrix of double type (was 'float2x2' (aka 'matrix<float, 2, 2>'))}}
}

double dxil_fma_bad_second(double a, float b, double c) {
  return fma(a, b, c);
  // dxil-error@-1 {{all arguments to 'fma' must have the same type}}
}

double dxil_fma_bad_third(double a, double b, half c) {
  return fma(a, b, c);
  // dxil-error@-1 {{all arguments to 'fma' must have the same type}}
}

double2 dxil_fma_bad_second_vec(double2 a, float2 b, double2 c) {
  return fma(a, b, c);
  // dxil-error@-1 {{all arguments to 'fma' must have the same type}}
}

double2x2 dxil_fma_bad_third_mat(double2x2 a, double2x2 b, float2x2 c) {
  return fma(a, b, c);
  // dxil-error@-1 {{all arguments to 'fma' must have the same type}}
}

double2 dxil_fma_mismatch_second(double2 a, double b, double2 c) {
  return fma(a, b, c);
  // dxil-error@-1 {{all arguments to 'fma' must have the same type}}
}

double2 dxil_fma_mismatch_third(double2 a, double2 b, double c) {
  return fma(a, b, c);
  // dxil-error@-1 {{all arguments to 'fma' must have the same type}}
}

double2x2 dxil_fma_mismatch_second_mat(double2x2 a, double2 b, double2x2 c) {
  return fma(a, b, c);
  // dxil-error@-1 {{all arguments to 'fma' must have the same type}}
}

double2x2 dxil_fma_mismatch_third_mat(double2x2 a, double2x2 b, double2 c) {
  return fma(a, b, c);
  // dxil-error@-1 {{all arguments to 'fma' must have the same type}}
}

half dxil_fma_half(half a, half b, half c) {
  return fma(a, b, c);
  // dxil-error@-1 {{1st argument must be a scalar, vector, or matrix of double type (was 'half')}}
}

half2 dxil_fma_half2(half2 a, half2 b, half2 c) {
  return fma(a, b, c);
  // dxil-error@-1 {{1st argument must be a scalar, vector, or matrix of double type (was 'half2' (aka 'vector<half, 2>'))}}
}

int dxil_fma_int(int a, int b, int c) {
  return fma(a, b, c);
  // dxil-error@-1 {{1st argument must be a scalar, vector, or matrix of double type (was 'int')}}
}

bool dxil_fma_bool(bool a, bool b, bool c) {
  return fma(a, b, c);
  // dxil-error@-1 {{1st argument must be a scalar, vector, or matrix of double type (was 'bool')}}
}
#endif

#ifdef TEST_SPIRV
int spv_fma_int(int a, int b, int c) {
  return fma(a, b, c);
  // spv-error@-1 {{1st argument must be a scalar or vector of floating-point type (was 'int')}}
}

int2 spv_fma_int2(int2 a, int2 b, int2 c) {
  return fma(a, b, c);
  // spv-error@-1 {{1st argument must be a scalar or vector of floating-point type (was 'int2' (aka 'vector<int, 2>'))}}
}

bool spv_fma_bool(bool a, bool b, bool c) {
  return fma(a, b, c);
  // spv-error@-1 {{1st argument must be a scalar or vector of floating-point type (was 'bool')}}
}

float spv_fma_bad_second(float a, int b, float c) {
  return fma(a, b, c);
  // spv-error@-1 {{all arguments to 'fma' must have the same type}}
}

float spv_fma_bad_third(float a, float b, bool c) {
  return fma(a, b, c);
  // spv-error@-1 {{all arguments to 'fma' must have the same type}}
}

float2 spv_fma_bad_second_vec(float2 a, int2 b, float2 c) {
  return fma(a, b, c);
  // spv-error@-1 {{all arguments to 'fma' must have the same type}}
}

double2 spv_fma_bad_third_vec(double2 a, double2 b, int2 c) {
  return fma(a, b, c);
  // spv-error@-1 {{all arguments to 'fma' must have the same type}}
}

float2 spv_fma_mismatch_second(float2 a, float b, float2 c) {
  return fma(a, b, c);
  // spv-error@-1 {{all arguments to 'fma' must have the same type}}
}

float2 spv_fma_mismatch_third(float2 a, float2 b, float c) {
  return fma(a, b, c);
  // spv-error@-1 {{all arguments to 'fma' must have the same type}}
}

double2 spv_fma_mismatch_second_double(double2 a, double b, double2 c) {
  return fma(a, b, c);
  // spv-error@-1 {{all arguments to 'fma' must have the same type}}
}

double2 spv_fma_mismatch_third_double(double2 a, double2 b, double c) {
  return fma(a, b, c);
  // spv-error@-1 {{all arguments to 'fma' must have the same type}}
}

float2x2 spv_fma_float2x2(float2x2 a, float2x2 b, float2x2 c) {
  return fma(a, b, c);
  // spv-error@-1 {{1st argument must be a scalar or vector of floating-point type (was 'float2x2' (aka 'matrix<float, 2, 2>'))}}
}

float2 spv_fma_bad_second_mat(float2 a, float2x2 b, float2 c) {
  return fma(a, b, c);
  // spv-error@-1 {{all arguments to 'fma' must have the same type}}
}

double2 spv_fma_bad_third_mat(double2 a, double2 b, double2x2 c) {
  return fma(a, b, c);
  // spv-error@-1 {{all arguments to 'fma' must have the same type}}
}

float2x3 spv_fma_float2x3(float2x3 a, float2x3 b, float2x3 c) {
  return fma(a, b, c);
  // spv-error@-1 {{1st argument must be a scalar or vector of floating-point type (was 'float2x3' (aka 'matrix<float, 2, 3>'))}}
}

float3x2 spv_fma_float3x2(float3x2 a, float3x2 b, float3x2 c) {
  return fma(a, b, c);
  // spv-error@-1 {{1st argument must be a scalar or vector of floating-point type (was 'float3x2' (aka 'matrix<float, 3, 2>'))}}
}

float4x4 spv_fma_float4x4(float4x4 a, float4x4 b, float4x4 c) {
  return fma(a, b, c);
  // spv-error@-1 {{1st argument must be a scalar or vector of floating-point type (was 'float4x4' (aka 'matrix<float, 4, 4>'))}}
}

double2x2 spv_fma_double2x2(double2x2 a, double2x2 b, double2x2 c) {
  return fma(a, b, c);
  // spv-error@-1 {{1st argument must be a scalar or vector of floating-point type (was 'double2x2' (aka 'matrix<double, 2, 2>'))}}
}

half2x2 spv_fma_half2x2(half2x2 a, half2x2 b, half2x2 c) {
  return fma(a, b, c);
  // spv-error@-1 {{1st argument must be a scalar or vector of floating-point type (was 'half2x2' (aka 'matrix<half, 2, 2>'))}}
}
#endif

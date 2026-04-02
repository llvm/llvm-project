// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.6-library %s -fnative-int16-type -emit-llvm-only -disable-llvm-passes -verify

float builtin_f16tof32_too_few_arg() {
  return __builtin_hlsl_elementwise_f16tof32();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
  // expected-note@hlsl/hlsl_alias_intrinsics.h:* 4 {{candidate function not viable: requires 1 argument, but 0 were provided}}
}

float builtin_f16tof32_too_many_arg(uint p0) {
  return __builtin_hlsl_elementwise_f16tof32(p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
  // expected-note@hlsl/hlsl_alias_intrinsics.h:* 4 {{candidate function not viable: requires 1 argument, but 2 were provided}}
}

float builtin_f16tof32_bool(bool p0) {
  return __builtin_hlsl_elementwise_f16tof32(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of unsigned integer types (was 'bool')}}
}

float builtin_f16tof32_bool4(bool4 p0) {
  return __builtin_hlsl_elementwise_f16tof32(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of unsigned integer types (was 'bool4' (aka 'vector<bool, 4>')}}
}

float builtin_f16tof32_short(short p0) {
  return __builtin_hlsl_elementwise_f16tof32(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of unsigned integer types (was 'short')}}
}

float builtin_f16tof32_unsigned_short(unsigned short p0) {
  return __builtin_hlsl_elementwise_f16tof32(p0);
  // expected-error@-1 {{incorrect number of bits in integer (expected 32 bits, have 16)}}
}

float builtin_f16tof32_int(int p0) {
  return __builtin_hlsl_elementwise_f16tof32(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of unsigned integer types (was 'int')}}
}

float builtin_f16tof32_int64_t(long p0) {
  return __builtin_hlsl_elementwise_f16tof32(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of unsigned integer types (was 'long')}}
}

float2 builtin_f16tof32_int2_to_float2_promotion(int2 p0) {
  return __builtin_hlsl_elementwise_f16tof32(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of unsigned integer types (was 'int2' (aka 'vector<int, 2>'))}}
}

float builtin_f16tof32_half(half p0) {
  return __builtin_hlsl_elementwise_f16tof32(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of unsigned integer types (was 'half')}}
}

float builtin_f16tof32_half4(half4 p0) {
  return __builtin_hlsl_elementwise_f16tof32(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of unsigned integer types (was 'half4' (aka 'vector<half, 4>'))}}
}

float builtin_f16tof32_float(float p0) {
  return __builtin_hlsl_elementwise_f16tof32(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of unsigned integer types (was 'float')}}
}

float builtin_f16tof32_double(double p0) {
  return __builtin_hlsl_elementwise_f16tof32(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of unsigned integer types (was 'double')}}
}

float f16tof32_too_few_arg() {
  return f16tof32();
  // expected-error@-1 {{no matching function for call to 'f16tof32'}}
}

float f16tof32_too_many_arg(uint p0) {
  return f16tof32(p0, p0);
  // expected-error@-1 {{no matching function for call to 'f16tof32'}}
}

float f16tof32_bool(bool p0) {
  return f16tof32(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of unsigned integer types (was 'bool')}}
}

float f16tof32_bool3(bool3 p0) {
  return f16tof32(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of unsigned integer types (was 'bool3' (aka 'vector<bool, 3>'))}}
}


float f16tof32_int16_t(short p0) {
  return f16tof32(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of unsigned integer types (was 'short')}}
}

float f16tof32_int16_t(unsigned short p0) {
  return f16tof32(p0);
  // expected-error@-1 {{incorrect number of bits in integer (expected 32 bits, have 16)}}
}

float f16tof32_int(int p0) {
  return f16tof32(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of unsigned integer types (was 'int')}}
}

float f16tof32_int64_t(long p0) {
  return f16tof32(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of unsigned integer types (was 'long')}}
}

float2 f16tof32_int2_to_float2_promotion(int3 p0) {
  return f16tof32(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of unsigned integer types (was 'int3' (aka 'vector<int, 3>'))}}
}

float f16tof32_half(half p0) {
  return f16tof32(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of unsigned integer types (was 'half')}}
}

float f16tof32_half2(half2 p0) {
  return f16tof32(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of unsigned integer types (was 'half2' (aka 'vector<half, 2>'))}}
}

float f16tof32_float(float p0) {
  return f16tof32(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of unsigned integer types (was 'float')}}
}

float f16tof32_double(double p0) {
  return f16tof32(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of unsigned integer types (was 'double')}}
}

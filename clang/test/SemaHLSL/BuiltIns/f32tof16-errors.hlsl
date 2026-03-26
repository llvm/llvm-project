// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.6-library %s -fnative-int16-type -emit-llvm-only -disable-llvm-passes -verify

uint builtin_f32tof16_too_few_arg() {
  return __builtin_hlsl_elementwise_f32tof16();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
  // expected-note@hlsl/hlsl_alias_intrinsics.h:* 4 {{candidate function not viable: requires 1 argument, but 0 were provided}}
}

uint builtin_f32tof16_too_many_arg(uint p0) {
  return __builtin_hlsl_elementwise_f32tof16(p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
  // expected-note@hlsl/hlsl_alias_intrinsics.h:* 4 {{candidate function not viable: requires 1 argument, but 2 were provided}}
}

uint builtin_f32tof16_bool(bool p0) {
  return __builtin_hlsl_elementwise_f32tof16(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'bool')}}
}

uint builtin_f32tof16_bool4(bool4 p0) {
  return __builtin_hlsl_elementwise_f32tof16(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'bool4' (aka 'vector<bool, 4>')}}
}

uint builtin_f32tof16_short(short p0) {
  return __builtin_hlsl_elementwise_f32tof16(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'short')}}
}

uint builtin_f32tof16_unsigned_short(unsigned short p0) {
  return __builtin_hlsl_elementwise_f32tof16(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned short')}}
}

uint builtin_f32tof16_int(int p0) {
  return __builtin_hlsl_elementwise_f32tof16(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}
}

uint builtin_f32tof16_int64_t(long p0) {
  return __builtin_hlsl_elementwise_f32tof16(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'long')}}
}

uint2 builtin_f32tof16_int2_to_float2_promotion(int2 p0) {
  return __builtin_hlsl_elementwise_f32tof16(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int2' (aka 'vector<int, 2>'))}}
}

uint builtin_f32tof16_half(half p0) {
  return __builtin_hlsl_elementwise_f32tof16(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'half')}}
}

uint builtin_f32tof16_half4(half4 p0) {
  return __builtin_hlsl_elementwise_f32tof16(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'half4' (aka 'vector<half, 4>'))}}
}

uint builtin_f32tof16_float(unsigned int p0) {
  return __builtin_hlsl_elementwise_f32tof16(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned int')}}
}

uint builtin_f32tof16_double(double p0) {
  return __builtin_hlsl_elementwise_f32tof16(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'double')}}
}

uint f32tof16_too_few_arg() {
  return f32tof16();
  // expected-error@-1 {{no matching function for call to 'f32tof16'}}
}

uint f32tof16_too_many_arg(uint p0) {
  return f32tof16(p0, p0);
  // expected-error@-1 {{no matching function for call to 'f32tof16'}}
}

uint f32tof16_bool(bool p0) {
  return f32tof16(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'bool')}}
}

uint f32tof16_bool3(bool3 p0) {
  return f32tof16(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'bool3' (aka 'vector<bool, 3>'))}}
}


uint f32tof16_int16_t(short p0) {
  return f32tof16(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'short')}}
}

uint f32tof16_int16_t(unsigned short p0) {
  return f32tof16(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'unsigned short')}}
}

uint f32tof16_int(int p0) {
  return f32tof16(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}
}

uint f32tof16_int64_t(long p0) {
  return f32tof16(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'long')}}
}

uint2 f32tof16_int2_to_float2_promotion(int3 p0) {
  return f32tof16(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int3' (aka 'vector<int, 3>'))}}
}

uint f32tof16_half(half p0) {
  return f32tof16(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'half')}}
}

uint f32tof16_half2(half2 p0) {
  return f32tof16(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'half2' (aka 'vector<half, 2>'))}}
}

uint f32tof16_float(uint p0) {
  return f32tof16(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'uint' (aka 'unsigned int'))}}
}

uint f32tof16_double(double p0) {
  return f32tof16(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'double')}}
}

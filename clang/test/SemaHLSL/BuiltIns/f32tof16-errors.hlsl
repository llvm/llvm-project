// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.6-library %s -fnative-int16-type -emit-llvm-only -disable-llvm-passes -verify

uint builtin_f32tof16_too_few_arg() {
  return __builtin_hlsl_elementwise_f32tof16();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
  // expected-note@hlsl/hlsl_alias_intrinsics_gen.inc:* 4 {{candidate function not viable: requires 1 argument, but 0 were provided}}
}

uint builtin_f32tof16_too_many_arg(uint p0) {
  return __builtin_hlsl_elementwise_f32tof16(p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
  // expected-note@hlsl/hlsl_alias_intrinsics_gen.inc:* 4 {{candidate function not viable: requires 1 argument, but 2 were provided}}
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

// Overload resolution selects uint f32tof16(float); bool is implicitly converted to float.
uint f32tof16_bool(bool p0) {
  return f32tof16(p0);
}

// Overload resolution selects an overload; bool3 is implicitly converted.
uint f32tof16_bool3(bool3 p0) {
  return f32tof16(p0);
  // expected-warning@-1 {{implicit conversion turns vector to scalar: 'vector<unsigned int, 3>' (vector of 3 'unsigned int' values) to 'unsigned int'}}
}


// Overload resolution selects uint f32tof16(float); short is implicitly converted to float.
uint f32tof16_int16_t(short p0) {
  return f32tof16(p0);
}

// Overload resolution selects uint f32tof16(float); unsigned short is implicitly converted to float.
uint f32tof16_int16_t(unsigned short p0) {
  return f32tof16(p0);
}

// Overload resolution selects uint f32tof16(float); int is implicitly converted to float.
uint f32tof16_int(int p0) {
  return f32tof16(p0);
  // expected-warning@-1 {{implicit conversion from 'int' to 'float' may lose precision}}
}

// Overload resolution selects uint f32tof16(float); long is implicitly converted to float.
uint f32tof16_int64_t(long p0) {
  return f32tof16(p0);
  // expected-warning@-1 {{implicit conversion from 'long' to 'float' may lose precision}}
}

// Overload resolution selects an overload; int3 is implicitly converted.
uint2 f32tof16_int2_to_float2_promotion(int3 p0) {
  return f32tof16(p0);
  // expected-warning@-1 {{implicit conversion truncates vector: 'vector<unsigned int, 3>' (vector of 3 'unsigned int' values) to 'vector<unsigned int, 2>' (vector of 2 'unsigned int' values)}}
  // expected-warning@-2 {{implicit conversion from 'int3' (aka 'vector<int, 3>') to 'vector<float, 3>' (vector of 3 'float' values) may lose precision}}
}

// Overload resolution selects uint f32tof16(float); half is implicitly converted to float.
uint f32tof16_half(half p0) {
  return f32tof16(p0);
}

// Overload resolution selects an overload; half2 is implicitly converted.
uint f32tof16_half2(half2 p0) {
  return f32tof16(p0);
  // expected-warning@-1 {{implicit conversion turns vector to scalar: 'vector<unsigned int, 2>' (vector of 2 'unsigned int' values) to 'unsigned int'}}
}

// Overload resolution selects uint f32tof16(float); uint is implicitly converted to float.
uint f32tof16_float(uint p0) {
  return f32tof16(p0);
  // expected-warning@-1 {{implicit conversion from 'uint' (aka 'unsigned int') to 'float' may lose precision}}
}

// Overload resolution selects uint f32tof16(float); double is implicitly converted to float.
uint f32tof16_double(double p0) {
  return f32tof16(p0);
  // expected-warning@-1 {{implicit conversion loses floating-point precision: 'double' to 'float'}}
}

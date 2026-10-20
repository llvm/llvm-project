// RUN: %clang_cc1 -fnative-half-type -fnative-int16-type -triple dxil-pc-shadermodel6.3-library -finclude-default-header -x hlsl -verify -o - %s
// RUN: %clang_cc1 -fnative-half-type -fnative-int16-type -triple spirv-pc-vulkan1.3-library -finclude-default-header -x hlsl -verify -o - %s

// Type and index checks for SV_Coverage, a scalar 32-bit int or uint, both as
// a pixel input and as a pixel output. These are independent of the shader
// stage and IO direction, which are covered by coverage.stages.hlsl.

[shader("pixel")]
float4 bad_input_float(float c : SV_Coverage) : SV_Target {
// expected-error@-1 {{semantic 'SV_Coverage' must be a scalar of 32 bit integer type (was 'float')}}
  return 0;
}

[shader("pixel")]
float4 ok_input_signed(int c : SV_Coverage) : SV_Target {
  return 0;
}

[shader("pixel")]
float4 bad_input_bool(bool c : SV_Coverage) : SV_Target {
// expected-error@-1 {{semantic 'SV_Coverage' must be a scalar of 32 bit integer type (was 'bool')}}
  return 0;
}

[shader("pixel")]
float4 bad_input_u16(uint16_t c : SV_Coverage) : SV_Target {
// expected-error@-1 {{semantic 'SV_Coverage' must be a scalar of 32 bit integer type (was 'uint16_t' (aka 'unsigned short'))}}
  return 0;
}

[shader("pixel")]
float4 bad_input_u64(uint64_t c : SV_Coverage) : SV_Target {
// expected-error@-1 {{semantic 'SV_Coverage' must be a scalar of 32 bit integer type (was 'uint64_t' (aka 'unsigned long'))}}
  return 0;
}

[shader("pixel")]
float4 bad_input_vector(uint2 c : SV_Coverage) : SV_Target {
// expected-error@-1 {{semantic 'SV_Coverage' must be a scalar of 32 bit integer type (was 'uint2' (aka 'vector<uint, 2>'))}}
  return 0;
}

[shader("pixel")]
float4 bad_input_vector_one(uint1 c : SV_Coverage) : SV_Target {
// expected-error@-1 {{semantic 'SV_Coverage' must be a scalar of 32 bit integer type (was 'uint1' (aka 'vector<uint, 1>'))}}
  return 0;
}

[shader("pixel")]
float4 bad_input_matrix(uint1x1 c : SV_Coverage) : SV_Target {
// expected-error@-1 {{semantic 'SV_Coverage' must be a scalar of 32 bit integer type (was 'uint1x1' (aka 'matrix<uint, 1, 1>'))}}
  return 0;
}

[shader("pixel")]
float bad_output_float(float4 p : SV_Position) : SV_Coverage {
// expected-error@-1 {{semantic 'SV_Coverage' must be a scalar of 32 bit integer type (was 'float')}}
  return p.x;
}

[shader("pixel")]
int ok_output_signed(float4 p : SV_Position) : SV_Coverage {
  return 0;
}

[shader("pixel")]
uint2 bad_output_vector(float4 p : SV_Position) : SV_Coverage {
// expected-error@-1 {{semantic 'SV_Coverage' must be a scalar of 32 bit integer type (was 'uint2' (aka 'vector<uint, 2>'))}}
  return 0;
}

struct BadOutput {
  float4 Color : SV_Target;
  half Mask : SV_Coverage;
// expected-error@-1 {{semantic 'SV_Coverage' must be a scalar of 32 bit integer type (was 'half')}}
};

[shader("pixel")]
BadOutput bad_output_struct(float4 p : SV_Position) {
  BadOutput o = { p, 0 };
  return o;
}

// SV_Coverage cannot be indexed, so arrays and non-zero indices are rejected.
[shader("pixel")]
float4 bad_input_index(uint c : SV_Coverage1) : SV_Target {
// expected-error@-1 {{semantic 'SV_Coverage' does not allow indexing}}
  return 0;
}

[shader("pixel")]
float4 bad_input_array(uint c[2] : SV_Coverage) : SV_Target {
// expected-error@-1 {{semantic 'SV_Coverage' does not allow indexing}}
// expected-error@-2 {{semantic 'SV_Coverage' must be a scalar of 32 bit integer type (was 'uint[2]'}}
  return 0;
}

[shader("pixel")]
float4 ok_input(uint c : SV_Coverage) : SV_Target {
  return 0;
}

[shader("pixel")]
uint ok_input_and_output(uint c : SV_Coverage) : SV_Coverage {
  return c;
}

[shader("pixel")]
uint ok_output_u32(float4 p : SV_Position) : SV_Coverage {
  return 1;
}

[shader("pixel")]
uint32_t ok_output_index(float4 p : SV_Position) : SV_Coverage0 {
  return 1;
}

struct Output {
  float4 Color : SV_Target;
  uint Mask : SV_Coverage;
};

[shader("pixel")]
Output ok_output_struct(uint c : SV_Coverage) {
  Output o = { float4(1, 1, 1, 1), c };
  return o;
}

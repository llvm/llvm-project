// RUN: %clang_cc1 -fnative-half-type -fnative-int16-type -triple dxil-pc-shadermodel6.2-library -finclude-default-header -x hlsl -verify -o - %s
// RUN: %clang_cc1 -fnative-half-type -fnative-int16-type -triple spirv-pc-vulkan1.3-library -finclude-default-header -x hlsl -verify -o - %s

// Type and index checks for SV_InnerCoverage, a scalar 32-bit int or uint.
// These are independent of the shader stage and IO direction, which are
// covered by innercoverage.stages.hlsl.

[shader("pixel")]
float4 bad_type_float(float c : SV_InnerCoverage) : SV_Target {
// expected-error@-1 {{semantic 'SV_InnerCoverage' must be a scalar of 32 bit integer type (was 'float')}}
  return 0;
}

[shader("pixel")]
float4 ok_signed(int c : SV_InnerCoverage) : SV_Target {
  return 0;
}

[shader("pixel")]
float4 bad_type_bool(bool c : SV_InnerCoverage) : SV_Target {
// expected-error@-1 {{semantic 'SV_InnerCoverage' must be a scalar of 32 bit integer type (was 'bool')}}
  return 0;
}

[shader("pixel")]
float4 bad_type_u16(uint16_t c : SV_InnerCoverage) : SV_Target {
// expected-error@-1 {{semantic 'SV_InnerCoverage' must be a scalar of 32 bit integer type (was 'uint16_t' (aka 'unsigned short'))}}
  return 0;
}

[shader("pixel")]
float4 bad_type_u64(uint64_t c : SV_InnerCoverage) : SV_Target {
// expected-error@-1 {{semantic 'SV_InnerCoverage' must be a scalar of 32 bit integer type (was 'uint64_t' (aka 'unsigned long'))}}
  return 0;
}

[shader("pixel")]
float4 bad_type_vector(uint2 c : SV_InnerCoverage) : SV_Target {
// expected-error@-1 {{semantic 'SV_InnerCoverage' must be a scalar of 32 bit integer type (was 'uint2' (aka 'vector<uint, 2>'))}}
  return 0;
}

[shader("pixel")]
float4 bad_type_vector_one(uint1 c : SV_InnerCoverage) : SV_Target {
// expected-error@-1 {{semantic 'SV_InnerCoverage' must be a scalar of 32 bit integer type (was 'uint1' (aka 'vector<uint, 1>'))}}
  return 0;
}

[shader("pixel")]
float4 bad_type_matrix(uint1x1 c : SV_InnerCoverage) : SV_Target {
// expected-error@-1 {{semantic 'SV_InnerCoverage' must be a scalar of 32 bit integer type (was 'uint1x1' (aka 'matrix<uint, 1, 1>'))}}
  return 0;
}

struct BadInput {
  float4 Pos : SV_Position;
  half Inner : SV_InnerCoverage;
// expected-error@-1 {{semantic 'SV_InnerCoverage' must be a scalar of 32 bit integer type (was 'half')}}
};

[shader("pixel")]
float4 bad_type_struct(BadInput In) : SV_Target {
  return 0;
}

// SV_InnerCoverage cannot be indexed, so arrays and non-zero indices are
// rejected.
[shader("pixel")]
float4 bad_index(uint c : SV_InnerCoverage1) : SV_Target {
// expected-error@-1 {{semantic 'SV_InnerCoverage' does not allow indexing}}
  return 0;
}

[shader("pixel")]
float4 bad_array(uint c[2] : SV_InnerCoverage) : SV_Target {
// expected-error@-1 {{semantic 'SV_InnerCoverage' does not allow indexing}}
// expected-error@-2 {{semantic 'SV_InnerCoverage' must be a scalar of 32 bit integer type (was 'uint[2]'}}
  return 0;
}

// The coverage and the inner coverage inputs cannot be read together.
[shader("pixel")]
float4 bad_with_coverage(uint c : SV_Coverage, uint ic : SV_InnerCoverage) : SV_Target {
// expected-error@-1 {{semantics 'SV_Coverage' and 'SV_InnerCoverage' are mutually exclusive in pixel shader inputs}}
// expected-note@-2 {{previous use is here}}
  return 0;
}

struct CoverageInputs {
  uint Inner : SV_InnerCoverage;
// expected-error@-1 {{semantics 'SV_Coverage' and 'SV_InnerCoverage' are mutually exclusive in pixel shader inputs}}
  float4 Pos : SV_Position;
  uint Coverage : SV_Coverage;
// expected-note@-1 {{previous use is here}}
};

[shader("pixel")]
float4 bad_with_coverage_struct(CoverageInputs In) : SV_Target {
  return 0;
}

// Writing the coverage output is still allowed with the inner coverage input.
[shader("pixel")]
uint ok_with_coverage_output(uint ic : SV_InnerCoverage) : SV_Coverage {
  return ic;
}

[shader("pixel")]
float4 ok(uint ic : SV_InnerCoverage0) : SV_Target {
  return 0;
}

struct Input {
  float4 Pos : SV_Position;
  uint32_t Inner : SV_InnerCoverage;
};

[shader("pixel")]
float4 ok_struct(Input In) : SV_Target {
  return 0;
}

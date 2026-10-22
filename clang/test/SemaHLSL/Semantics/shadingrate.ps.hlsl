// RUN: %clang_cc1 -fnative-half-type -fnative-int16-type -triple dxil-pc-shadermodel6.4-library -finclude-default-header -x hlsl -verify=dxil,expected -o - %s
// RUN: %clang_cc1 -fnative-half-type -fnative-int16-type -triple spirv-pc-vulkan1.3-library -finclude-default-header -x hlsl -verify=spirv,expected -o - %s

// Type and index checks for SV_ShadingRate. These are independent of the
// shader stage and IO direction, which are covered by shadingrate.stages.hlsl.

[shader("pixel")]
float4 bad_type_float(float r : SV_ShadingRate) : SV_Target {
// dxil-error@-1 {{semantic 'SV_ShadingRate' must be a scalar of 16 or 32 bit unsigned integer type (was 'float')}}
// spirv-error@-2 {{semantic 'SV_ShadingRate' must be a scalar of 32 bit unsigned integer type (was 'float')}}
  return float4((float)r, 0, 0, 1);
}

[shader("pixel")]
float4 bad_type_signed(int r : SV_ShadingRate) : SV_Target {
// dxil-error@-1 {{semantic 'SV_ShadingRate' must be a scalar of 16 or 32 bit unsigned integer type (was 'int')}}
// spirv-error@-2 {{semantic 'SV_ShadingRate' must be a scalar of 32 bit unsigned integer type (was 'int')}}
  return float4((float)r, 0, 0, 1);
}

[shader("pixel")]
float4 bad_type_u64(uint64_t r : SV_ShadingRate) : SV_Target {
// dxil-error@-1 {{semantic 'SV_ShadingRate' must be a scalar of 16 or 32 bit unsigned integer type (was 'uint64_t' (aka 'unsigned long'))}}
// spirv-error@-2 {{semantic 'SV_ShadingRate' must be a scalar of 32 bit unsigned integer type (was 'uint64_t' (aka 'unsigned long'))}}
  return float4((float)r, 0, 0, 1);
}

[shader("pixel")]
float4 bad_type_vector(uint3 r : SV_ShadingRate) : SV_Target {
// dxil-error@-1 {{semantic 'SV_ShadingRate' must be a scalar of 16 or 32 bit unsigned integer type (was 'uint3' (aka 'vector<uint, 3>'))}}
// spirv-error@-2 {{semantic 'SV_ShadingRate' must be a scalar of 32 bit unsigned integer type (was 'uint3' (aka 'vector<uint, 3>'))}}
  return float4((float)r.x, 0, 0, 1);
}

[shader("pixel")]
float4 ok_u32(uint32_t r : SV_ShadingRate) : SV_Target {
  return float4((float)r, 0, 0, 1);
}

// DXIL permits U16, SPIR-V requires a scalar 32-bit integer.
[shader("pixel")]
float4 ok_dxil_bad_spirv(uint16_t r : SV_ShadingRate) : SV_Target {
// spirv-error@-1 {{semantic 'SV_ShadingRate' must be a scalar of 32 bit unsigned integer type (was 'uint16_t' (aka 'unsigned short'))}}
  return float4((float)r, 0, 0, 1);
}

[shader("pixel")]
float4 bad_index(uint r : SV_ShadingRate1) : SV_Target {
// expected-error@-1 {{semantic 'SV_ShadingRate' does not allow indexing}}
  return float4((float)r, 0, 0, 1);
}

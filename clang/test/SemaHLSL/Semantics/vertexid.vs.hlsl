// RUN: %clang_cc1 -fnative-half-type -fnative-int16-type -triple dxil-pc-shadermodel6.3-library -finclude-default-header -x hlsl -verify -o - %s
// RUN: %clang_cc1 -fnative-half-type -fnative-int16-type -triple spirv-pc-vulkan1.3-library -finclude-default-header -x hlsl -verify -o - %s

// SV_VertexID accepts 16- and 32-bit signed or unsigned scalars.

[shader("vertex")]
float bad_type_float(float id : SV_VertexID) : A {
// expected-error@-1 {{semantic 'SV_VertexID' must be a scalar of 16 or 32 bit integer type (was 'float')}}
  return id;
}

[shader("vertex")]
uint3 bad_type_vector(uint3 id : SV_VertexID) : A {
// expected-error@-1 {{semantic 'SV_VertexID' must be a scalar of 16 or 32 bit integer type (was 'uint3' (aka 'vector<uint, 3>'))}}
  return id;
}

[shader("vertex")]
uint64_t bad_type_size(uint64_t id : SV_VertexID) : A {
// expected-error@-1 {{semantic 'SV_VertexID' must be a scalar of 16 or 32 bit integer type (was 'uint64_t' (aka 'unsigned long'))}}
  return id;
}

[shader("vertex")]
float bad_type_bool(bool id : SV_VertexID) : A {
// expected-error@-1 {{semantic 'SV_VertexID' must be a scalar of 16 or 32 bit integer type (was 'bool')}}
  return id;
}

[shader("vertex")]
uint32_t ok_unsigned(uint32_t id : SV_VertexID) : A {
  return id;
}

[shader("vertex")]
int ok_signed(int id : SV_VertexID) : A {
  return id;
}

[shader("vertex")]
uint16_t ok_16bit(uint16_t id : SV_VertexID) : A {
  return id;
}

// RUN: %clang_cc1 -fnative-half-type -fnative-int16-type -triple dxil-pc-shadermodel6.3-library -finclude-default-header -x hlsl -verify=dxil,expected -o - %s
// RUN: %clang_cc1 -fnative-half-type -fnative-int16-type -triple spirv-pc-vulkan1.3-library -finclude-default-header -x hlsl -verify=spirv,expected -o - %s

[shader("vertex")]
float bad_type_float(float id : SV_InstanceID) : A {
// dxil-error@-1 {{semantic 'SV_InstanceID' must be a scalar of 16 or 32 bit unsigned integer type (was 'float')}}
// spirv-error@-2 {{semantic 'SV_InstanceID' must be a scalar of 32 bit unsigned integer type (was 'float')}}
  return id;
}

[shader("vertex")]
uint3 bad_type_vector(uint3 id : SV_InstanceID) : A {
// dxil-error@-1 {{semantic 'SV_InstanceID' must be a scalar of 16 or 32 bit unsigned integer type (was 'uint3' (aka 'vector<uint, 3>'))}}
// spirv-error@-2 {{semantic 'SV_InstanceID' must be a scalar of 32 bit unsigned integer type (was 'uint3' (aka 'vector<uint, 3>'))}}
  return id;
}

[shader("vertex")]
uint bad_type_vector_one(uint1 id : SV_InstanceID) : A {
// dxil-error@-1 {{semantic 'SV_InstanceID' must be a scalar of 16 or 32 bit unsigned integer type (was 'uint1' (aka 'vector<uint, 1>'))}}
// spirv-error@-2 {{semantic 'SV_InstanceID' must be a scalar of 32 bit unsigned integer type (was 'uint1' (aka 'vector<uint, 1>'))}}
  return id.x;
}

[shader("vertex")]
uint bad_type_matrix(uint1x1 id : SV_InstanceID) : A {
// dxil-error@-1 {{semantic 'SV_InstanceID' must be a scalar of 16 or 32 bit unsigned integer type (was 'uint1x1' (aka 'matrix<uint, 1, 1>'))}}
// spirv-error@-2 {{semantic 'SV_InstanceID' must be a scalar of 32 bit unsigned integer type (was 'uint1x1' (aka 'matrix<uint, 1, 1>'))}}
  return 0;
}

[shader("vertex")]
int bad_type_signed(int id : SV_InstanceID) : A {
// dxil-error@-1 {{semantic 'SV_InstanceID' must be a scalar of 16 or 32 bit unsigned integer type (was 'int')}}
// spirv-error@-2 {{semantic 'SV_InstanceID' must be a scalar of 32 bit unsigned integer type (was 'int')}}
  return id;
}

[shader("vertex")]
uint bad_type_u64(uint64_t id : SV_InstanceID) : A {
// dxil-error@-1 {{semantic 'SV_InstanceID' must be a scalar of 16 or 32 bit unsigned integer type (was 'uint64_t' (aka 'unsigned long'))}}
// spirv-error@-2 {{semantic 'SV_InstanceID' must be a scalar of 32 bit unsigned integer type (was 'uint64_t' (aka 'unsigned long'))}}
  return (uint)id;
}

[shader("vertex")]
char bad_type_char(char id : SV_InstanceID) : A {
// dxil-error@-1 {{semantic 'SV_InstanceID' must be a scalar of 16 or 32 bit unsigned integer type (was 'char')}}
// spirv-error@-2 {{semantic 'SV_InstanceID' must be a scalar of 32 bit unsigned integer type (was 'char')}}
  return id;
}

[shader("vertex")]
uint bad_type_bool(bool id : SV_InstanceID) : A {
// dxil-error@-1 {{semantic 'SV_InstanceID' must be a scalar of 16 or 32 bit unsigned integer type (was 'bool')}}
// spirv-error@-2 {{semantic 'SV_InstanceID' must be a scalar of 32 bit unsigned integer type (was 'bool')}}
  return id;
}

[shader("vertex")]
uint32_t ok_u32(uint32_t id : SV_InstanceID) : A {
  return id;
}

// DXIL permits U16, SPIRV rejects it: VUID-InstanceIndex-InstanceIndex-04265
// requires a scalar 32-bit integer.
[shader("vertex")]
uint16_t ok_dxil_bad_spirv(uint16_t id : SV_InstanceID) : A {
// spirv-error@-1 {{semantic 'SV_InstanceID' must be a scalar of 32 bit unsigned integer type (was 'uint16_t' (aka 'unsigned short'))}}
  return id;
}

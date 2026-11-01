// RUN: %clang_cc1 -fnative-half-type -fnative-int16-type -triple dxil-pc-shadermodel6.3-library -finclude-default-header -x hlsl -verify=dxil,expected -o - %s
// RUN: %clang_cc1 -fnative-half-type -fnative-int16-type -triple spirv-pc-vulkan1.3-library -finclude-default-header -x hlsl -verify=spirv,expected -o - %s

float bad_type_float(float id : SV_InstanceID) : A {
// expected-error@-1 {{attribute 'SV_InstanceID' only applies to a field or parameter of type 'uint'}}
  return id;
}

uint3 bad_type_vector(uint3 id : SV_InstanceID) : A {
// expected-error@-1 {{attribute 'SV_InstanceID' only applies to a field or parameter of type 'uint'}}
  return id;
}

int bad_type_signed(int id : SV_InstanceID) : A {
// expected-error@-1 {{attribute 'SV_InstanceID' only applies to a field or parameter of type 'uint'}}
  return id;
}

uint64_t bad_type_u64(uint64_t id : SV_InstanceID) : A {
// expected-error@-1 {{attribute 'SV_InstanceID' only applies to a field or parameter of type 'uint'}}
  return id;
}

char bad_type_char(char id : SV_InstanceID) : A {
// expected-error@-1 {{attribute 'SV_InstanceID' only applies to a field or parameter of type 'uint'}}
  return id;
}

uint32_t ok_u32(uint32_t id : SV_InstanceID) : A {
  return id;
}

// DXIL permits U16, SPIRV rejects it: VUID-InstanceIndex-InstanceIndex-04265
// requires a scalar 32-bit integer.
uint16_t ok_dxil_bad_spirv(uint16_t id : SV_InstanceID) : A {
// spirv-error@-1 {{attribute 'SV_InstanceID' only applies to a field or parameter of type 'uint'}}
  return id;
}

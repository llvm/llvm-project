// RUN: %clang_cc1 -fnative-half-type -fnative-int16-type -triple dxil-pc-shadermodel6.3-vertex -finclude-default-header -x hlsl -verify -o - %s
// RUN: %clang_cc1 -fnative-half-type -fnative-int16-type -triple spirv-pc-vulkan1.3-vertex -finclude-default-header -x hlsl -verify -o - %s

float bad_type_float(float id : SV_VertexID) : A {
// expected-error@-1 {{attribute 'SV_VertexID' only applies to a field or parameter of type 'uint'}}
  return id;
}

uint3 bad_type_vector(uint3 id : SV_VertexID) : A {
// expected-error@-1 {{attribute 'SV_VertexID' only applies to a field or parameter of type 'uint'}}
  return id;
}

int bad_type_signed(int id : SV_VertexID) : A {
// expected-error@-1 {{attribute 'SV_VertexID' only applies to a field or parameter of type 'uint'}}
  return id;
}

uint64_t bad_type_size(uint64_t id : SV_VertexID) : A {
// expected-error@-1 {{attribute 'SV_VertexID' only applies to a field or parameter of type 'uint'}}
  return id;
}

uint32_t ok(uint32_t id : SV_VertexID) : A {
  return id;
}

uint16_t bad_type_size_2(uint16_t id : SV_VertexID) : A {
// expected-error@-1 {{attribute 'SV_VertexID' only applies to a field or parameter of type 'uint'}}
  return id;
}

char bad_type_size_2(char id : SV_VertexID) : A {
// expected-error@-1 {{attribute 'SV_VertexID' only applies to a field or parameter of type 'uint'}}
  return id;
}

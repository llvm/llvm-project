// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-library -fnative-half-type -fnative-int16-type -finclude-default-header -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -fnative-half-type -fnative-int16-type -finclude-default-header -fsyntax-only -verify %s

// Valid type, index and interpolation uses of SV_PrimitiveID. The invalid
// cases are covered by primitiveid.ps.errors.hlsl, and the shader stage and IO
// direction by primitiveid.stages.hlsl.

// expected-no-diagnostics

[shader("pixel")]
void valid_uint(uint s : SV_PrimitiveID) {}
[shader("pixel")]
void valid_int(int s : sv_primitiveid0) {}
[shader("pixel")]
void valid_flat(nointerpolation uint s : SV_PrimitiveID) {}

// 16-bit integers are allowed. SPIR-V converts them to and from the 32-bit
// built-in.
[shader("pixel")]
void uint16(uint16_t s : SV_PrimitiveID) {}
[shader("pixel")]
void int16(int16_t s : SV_PrimitiveID) {}

struct Input {
  uint s : SV_PrimitiveID;
};
[shader("pixel")]
void valid_struct(Input i) {}

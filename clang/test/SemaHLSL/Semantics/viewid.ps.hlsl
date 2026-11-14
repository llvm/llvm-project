// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-library -fnative-half-type -fnative-int16-type -finclude-default-header -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -fnative-half-type -fnative-int16-type -finclude-default-header -fsyntax-only -verify %s

// Valid type, index and interpolation uses of SV_ViewID. The invalid cases are
// covered by viewid.ps.errors.hlsl, and the shader stage and IO direction by
// viewid.stages.hlsl.

// expected-no-diagnostics

[shader("pixel")]
void valid_uint(uint v : SV_ViewID) {}
[shader("pixel")]
void valid_int(int v : sv_viewid0) {}
[shader("pixel")]
void valid_flat(nointerpolation uint v : SV_ViewID) {}

struct Input {
  float4 p : SV_Position;
  uint v : SV_ViewID;
};
[shader("pixel")]
void valid_struct(Input i) {}

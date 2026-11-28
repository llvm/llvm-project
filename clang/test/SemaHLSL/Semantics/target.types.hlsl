// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -fnative-half-type -finclude-default-header -x hlsl -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -fnative-half-type -finclude-default-header -x hlsl -fsyntax-only -verify %s

// Valid types of SV_Target: each render target is a scalar or vector of up to
// four 16- or 32-bit floats. Arrays cover consecutive render targets. The
// invalid cases are covered by target.types.errors.hlsl.

// expected-no-diagnostics

[shader("pixel")]
float scalar() : SV_Target { return 0; }

[shader("pixel")]
half2 half_vector() : SV_Target { return 0; }

struct Targets {
  float4 Color : SV_Target0;
  float3 Layers[2] : SV_Target1;
};

[shader("pixel")]
Targets array() { return (Targets)0; }

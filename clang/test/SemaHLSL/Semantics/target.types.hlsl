// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -fnative-half-type -finclude-default-header -x hlsl -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -fnative-half-type -finclude-default-header -x hlsl -fsyntax-only -verify %s

// SV_Target: each render target is a scalar or vector of up to four 16- or
// 32-bit floats. Arrays cover consecutive render targets.

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

[shader("pixel")]
int4 signed_int() : SV_Target { return 0; }
// expected-error@-1 {{semantic 'SV_Target' must be a scalar or vector of up to 4 components of 16 or 32 bit floating-point type (was 'int4' (aka 'vector<int, 4>'))}}

[shader("pixel")]
uint unsigned_int() : SV_Target { return 0; }
// expected-error@-1 {{semantic 'SV_Target' must be a scalar or vector of up to 4 components of 16 or 32 bit floating-point type (was 'uint' (aka 'unsigned int'))}}

[shader("pixel")]
bool boolean() : SV_Target { return 0; }
// expected-error@-1 {{semantic 'SV_Target' must be a scalar or vector of up to 4 components of 16 or 32 bit floating-point type (was 'bool')}}

[shader("pixel")]
float2x4 matrix() : SV_Target { return 0; }
// expected-error@-1 {{semantic 'SV_Target' must be a scalar or vector of up to 4 components of 16 or 32 bit floating-point type (was 'float2x4' (aka 'matrix<float, 2, 4>'))}}

[shader("pixel")]
vector<float, 5> too_wide() : SV_Target { return 0; }
// expected-error@-1 {{semantic 'SV_Target' must be a scalar or vector of up to 4 components of 16 or 32 bit floating-point type (was 'vector<float, 5>' (vector of 5 'float' values))}}

// An array uses the render targets of all its elements, so they must not
// overlap with another SV_Target.
struct Overlap {
  float4 Layers[2] : SV_Target0;
// expected-note@-1 {{previous use is here}}
  float4 Color : SV_Target1;
// expected-error@-1 {{semantic index overlap SV_Target1}}
// expected-note@-2 {{'Color' used here}}
};

[shader("pixel")]
Overlap overlap() { return (Overlap)0; }

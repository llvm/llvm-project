// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-library -fnative-half-type -fnative-int16-type -finclude-default-header -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -DSPIRV -fnative-half-type -fnative-int16-type -finclude-default-header -fsyntax-only -verify %s

// Type, index and interpolation checks for SV_Barycentrics. These are
// independent of the shader stage and IO direction, which are covered by
// barycentrics.stages.hlsl.

[shader("pixel")]
float4 valid(float3 p : SV_Barycentrics0) : SV_Target {
  return float4(p, 1);
}
[shader("pixel")]
void affine(noperspective float3 p : sv_barycentrics) {}
[shader("pixel")]
void centroid_input(centroid float3 p : SV_Barycentrics) {}
[shader("pixel")]
void sample_input(sample noperspective float3 p : SV_Barycentrics) {}
[shader("pixel")]
void half_vector(half3 b : SV_Barycentrics) {}

[shader("pixel")]
void scalar(float b : SV_Barycentrics) {}
// expected-error@-1 {{semantic 'SV_Barycentrics' must be a vector of 3 components of 16 or 32 bit floating-point type}}
[shader("pixel")]
void short_vector(float2 b : SV_Barycentrics) {}
// expected-error@-1 {{semantic 'SV_Barycentrics' must be a vector of 3 components of 16 or 32 bit floating-point type}}
[shader("pixel")]
void long_vector(float4 b : SV_Barycentrics) {}
// expected-error@-1 {{semantic 'SV_Barycentrics' must be a vector of 3 components of 16 or 32 bit floating-point type}}
[shader("pixel")]
void double_vector(double3 b : SV_Barycentrics) {}
// expected-error@-1 {{semantic 'SV_Barycentrics' must be a vector of 3 components of 16 or 32 bit floating-point type}}
[shader("pixel")]
void integer_vector(uint3 b : SV_Barycentrics) {}
// expected-error@-1 {{semantic 'SV_Barycentrics' must be a vector of 3 components of 16 or 32 bit floating-point type}}
[shader("pixel")]
void matrix(float1x3 b : SV_Barycentrics) {}
// expected-error@-1 {{semantic 'SV_Barycentrics' must be a vector of 3 components of 16 or 32 bit floating-point type}}
// expected-error@-2 {{semantic 'SV_Barycentrics' does not allow indexing}}
[shader("pixel")]
void array(float3 b[2] : SV_Barycentrics) {}
// expected-error@-1 {{semantic 'SV_Barycentrics' must be a vector of 3 components of 16 or 32 bit floating-point type}}
// expected-error@-2 {{semantic 'SV_Barycentrics' does not allow indexing}}

[shader("pixel")]
void bad_index_one(float3 b : SV_Barycentrics1) {}
// expected-error@-1 {{semantic 'SV_Barycentrics' does not allow indexing}}

[shader("pixel")]
void flat(nointerpolation float3 b : SV_Barycentrics) {} // expected-note {{conflicting attribute is here}}
// expected-error@-1 {{'nointerpolation' cannot be used on SV_Barycentrics}}

[shader("pixel")]
void same_index(float3 p : SV_Barycentrics, // expected-note {{previous use is here}}
                noperspective float3 q : SV_Barycentrics0) {} // expected-note {{'q' declared here}}
// expected-error@-1 {{semantic index overlap SV_Barycentrics0}}

struct Input {
  float3 b : SV_Barycentrics; // expected-note {{conflicting attribute is here}}
};
[shader("pixel")]
void inherited_flat(nointerpolation Input i) {}
// expected-error@-1 {{'nointerpolation' cannot be used on SV_Barycentrics}}

struct AffineInput {
  noperspective float3 b : SV_Barycentrics;
};
[shader("pixel")]
void inherited_affine(noperspective Input i) {}
[shader("pixel")]
void field_override(linear AffineInput i) {}

#ifdef SPIRV
[shader("pixel")]
float4 explicit_location(float3 b : SV_Barycentrics,
                         [[vk::location(3)]] float x : VALUE) : SV_Target {
  return float4(b, x);
}
#endif

// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -x hlsl \
// RUN:   -fsyntax-only -finclude-default-header -fnative-half-type \
// RUN:   -fnative-int16-type -DTEXTURE=Texture2D -DGRAD_TYPE=float2 \
// RUN:   -DCOORD_TYPE=float2 -verify=sm66,expected %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -x hlsl \
// RUN:   -fsyntax-only -finclude-default-header -fnative-half-type \
// RUN:   -fnative-int16-type -DTEXTURE=Texture2DArray -DGRAD_TYPE=float2 \
// RUN:   -DCOORD_TYPE=float3 -verify=sm66,expected %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.7-library -x hlsl \
// RUN:   -fsyntax-only -finclude-default-header -fnative-half-type \
// RUN:   -fnative-int16-type -DTEXTURE=Texture2D -DGRAD_TYPE=float2 \
// RUN:   -DCOORD_TYPE=float2 -verify=expected %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.7-library -x hlsl \
// RUN:   -fsyntax-only -finclude-default-header -fnative-half-type \
// RUN:   -fnative-int16-type -DTEXTURE=Texture2DArray -DGRAD_TYPE=float2 \
// RUN:   -DCOORD_TYPE=float3 -verify=expected %s
// RUN: %clang_cc1 -triple spirv-unknown-vulkan-library -x hlsl -fsyntax-only \
// RUN:   -finclude-default-header -fnative-half-type -fnative-int16-type \
// RUN:   -DTEXTURE=Texture2D -DGRAD_TYPE=float2 -DCOORD_TYPE=float2 \
// RUN:   -verify=expected %s

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   TEXTURE            resource type name
//   GRAD_TYPE          SampleGrad ddx/ddy type, one component per resource
//                      dimension
//   COORD_TYPE         sample location type (DIM components plus the array
//                      slice)
//
// Check prefixes:
//   sm66               shader model 6.6 diagnostics
//
// Sampling textures with an integer element type was introduced in shader model
// 6.7, so the `sm66` diagnostics are only expected in the shader model 6.6 runs.
// The Vulkan target has no shader model and does not restrict integer sampling.
// Comparison sampling requires a floating point element type at every shader
// model, so those diagnostics use the `expected` prefix.

TEXTURE<int4> TexVec;
TEXTURE<uint> Tex;
TEXTURE<int16_t> TexShort;
SamplerState Samp;
SamplerComparisonState SampCmp;

void main(COORD_TYPE uv) {
  float compare = 0.5f;

  // sm66-error@* {{'Sample' on resources containing 'unsigned int' requires shader model 6.7 or newer; the target shader model is 6.6}}
  // sm66-note-re@*:* {{in instantiation of member function 'hlsl::Texture{{.+}}<unsigned int>::Sample' requested here}}
  Tex.Sample(Samp, uv);

  // sm66-error@* {{'SampleBias' on resources containing 'unsigned int' requires shader model 6.7 or newer; the target shader model is 6.6}}
  // sm66-note-re@*:* {{in instantiation of member function 'hlsl::Texture{{.+}}<unsigned int>::SampleBias' requested here}}
  Tex.SampleBias(Samp, uv, 0.5f);

  // sm66-error@* {{'SampleGrad' on resources containing 'unsigned int' requires shader model 6.7 or newer; the target shader model is 6.6}}
  // sm66-note-re@*:* {{in instantiation of member function 'hlsl::Texture{{.+}}<unsigned int>::SampleGrad' requested here}}
  Tex.SampleGrad(Samp, uv, (GRAD_TYPE)0, (GRAD_TYPE)0);

  // sm66-error@* {{'SampleLevel' on resources containing 'unsigned int' requires shader model 6.7 or newer; the target shader model is 6.6}}
  // sm66-note-re@*:* {{in instantiation of member function 'hlsl::Texture{{.+}}<unsigned int>::SampleLevel' requested here}}
  Tex.SampleLevel(Samp, uv, 0.0f);

  // Textures containing vectors of integers and 16-bit integers are gated the
  // same way.
  // sm66-error@* {{'Sample' on resources containing 'vector<int, 4>' (vector of 4 'int' values) requires shader model 6.7 or newer; the target shader model is 6.6}}
  // sm66-note-re@*:* {{in instantiation of member function 'hlsl::Texture{{.+}}<vector<int, 4>>::Sample' requested here}}
  TexVec.Sample(Samp, uv);

  // sm66-error@* {{'SampleLevel' on resources containing 'short' requires shader model 6.7 or newer; the target shader model is 6.6}}
  // sm66-note-re@*:* {{in instantiation of member function 'hlsl::Texture{{.+}}<short>::SampleLevel' requested here}}
  TexShort.SampleLevel(Samp, uv, 0.0f);

  // Comparison sampling rejects integer element types at every shader model.
  // expected-error@* {{'SampleCmp' and 'SampleCmpLevelZero' require resource to contain a floating point type}}
  // expected-note-re@*:* {{in instantiation of member function 'hlsl::Texture{{.+}}<unsigned int>::SampleCmp' requested here}}
  Tex.SampleCmp(SampCmp, uv, compare);

  // expected-error@* {{'SampleCmp' and 'SampleCmpLevelZero' require resource to contain a floating point type}}
  // expected-note-re@*:* {{in instantiation of member function 'hlsl::Texture{{.+}}<unsigned int>::SampleCmpLevelZero' requested here}}
  Tex.SampleCmpLevelZero(SampCmp, uv, compare);

  // Gathering on textures with an integer element type is allowed at every
  // shader model.
  Tex.Gather(Samp, uv);
  TexVec.Gather(Samp, uv);
}

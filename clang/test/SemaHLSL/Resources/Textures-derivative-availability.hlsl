// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header \
// RUN:   -fsyntax-only -verify=expected,sm66 -DTEXTURE=Texture2D -DCOORD_TYPE=float2 %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.5-library -finclude-default-header \
// RUN:   -fsyntax-only -verify=expected,sm65 -DTEXTURE=Texture2D -DCOORD_TYPE=float2 %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header \
// RUN:   -fsyntax-only -verify=expected,sm66 -DTEXTURE=Texture2DArray -DCOORD_TYPE=float3 %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.5-library -finclude-default-header \
// RUN:   -fsyntax-only -verify=expected,sm65 -DTEXTURE=Texture2DArray -DCOORD_TYPE=float3 %s

// Texture methods that rely on implicit derivatives are available in pixel
// shaders since Shader Model 6.0 and in compute, mesh and amplification
// shaders since Shader Model 6.6. They are not available in any other shader
// stage.

TEXTURE<float4> tex;
SamplerState samp;
SamplerComparisonState cmpSamp;

// Derivatives are always available in pixel shaders; no diagnostics expected.
[shader("pixel")]
void PixelEntry() {
  COORD_TYPE loc = (COORD_TYPE)0;
  float2 lodLoc = float2(0, 0);

  tex.Sample(samp, loc);
  tex.SampleBias(samp, loc, 0.5);
  tex.SampleCmp(cmpSamp, loc, 0.5);
  tex.CalculateLevelOfDetail(samp, lodLoc);
  tex.CalculateLevelOfDetailUnclamped(samp, lodLoc);
}

// Derivatives are available in compute shaders only since Shader Model 6.6.
[shader("compute")]
[numthreads(1, 1, 1)]
void ComputeEntry() {
  COORD_TYPE loc = (COORD_TYPE)0;
  float2 lodLoc = float2(0, 0);

  // sm65-error@+2 {{'Sample' is only available in compute environment on Shader Model 6.6 or newer}}
  // sm65-note@* {{'Sample' has been marked as being introduced in Shader Model 6.6 in compute environment here, but the deployment target is Shader Model 6.5 compute environment}}
  tex.Sample(samp, loc);

  // sm65-error@+2 {{'SampleBias' is only available in compute environment on Shader Model 6.6 or newer}}
  // sm65-note@* {{'SampleBias' has been marked as being introduced in Shader Model 6.6 in compute environment here}}
  tex.SampleBias(samp, loc, 0.5);

  // sm65-error@+2 {{'SampleCmp' is only available in compute environment on Shader Model 6.6 or newer}}
  // sm65-note@* {{'SampleCmp' has been marked as being introduced in Shader Model 6.6 in compute environment here}}
  tex.SampleCmp(cmpSamp, loc, 0.5);

  // sm65-error@+2 {{'CalculateLevelOfDetail' is only available in compute environment on Shader Model 6.6 or newer}}
  // sm65-note@* {{'CalculateLevelOfDetail' has been marked as being introduced in Shader Model 6.6 in compute environment here}}
  tex.CalculateLevelOfDetail(samp, lodLoc);

  // sm65-error@+2 {{'CalculateLevelOfDetailUnclamped' is only available in compute environment on Shader Model 6.6 or newer}}
  // sm65-note@* {{'CalculateLevelOfDetailUnclamped' has been marked as being introduced in Shader Model 6.6 in compute environment here}}
  tex.CalculateLevelOfDetailUnclamped(samp, lodLoc);
}

// Vertex shaders do not support derivatives in any shader model.
[shader("vertex")]
void VertexEntry() {
  COORD_TYPE loc = (COORD_TYPE)0;
  float2 lodLoc = float2(0, 0);

  // expected-error@+2 {{'Sample' is unavailable}}
  // expected-note@* {{'Sample' has been marked as being introduced in Shader Model}}
  tex.Sample(samp, loc);

  // expected-error@+2 {{'SampleBias' is unavailable}}
  // expected-note@* {{'SampleBias' has been marked as being introduced in Shader Model}}
  tex.SampleBias(samp, loc, 0.5);

  // expected-error@+2 {{'SampleCmp' is unavailable}}
  // expected-note@* {{'SampleCmp' has been marked as being introduced in Shader Model}}
  tex.SampleCmp(cmpSamp, loc, 0.5);

  // expected-error@+2 {{'CalculateLevelOfDetail' is unavailable}}
  // expected-note@* {{'CalculateLevelOfDetail' has been marked as being introduced in Shader Model}}
  tex.CalculateLevelOfDetail(samp, lodLoc);

  // expected-error@+2 {{'CalculateLevelOfDetailUnclamped' is unavailable}}
  // expected-note@* {{'CalculateLevelOfDetailUnclamped' has been marked as being introduced in Shader Model}}
  tex.CalculateLevelOfDetailUnclamped(samp, lodLoc);
}

// Methods that take an explicit LOD or explicit gradients do not require
// derivatives and are available in all shader stages; no diagnostics expected.
[shader("vertex")]
void ExplicitLodVertexEntry() {
  COORD_TYPE loc = (COORD_TYPE)0;
  float2 grad = float2(0, 0);

  tex.SampleLevel(samp, loc, 0);
  tex.SampleGrad(samp, loc, grad, grad);
  tex.SampleCmpLevelZero(cmpSamp, loc, 0.5);
  tex.Gather(samp, loc);
}

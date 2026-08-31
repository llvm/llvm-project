// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library \
// RUN:   -finclude-default-header -fsyntax-only -verify=expected,dim2 \
// RUN:   -DTEXTURE=Texture2D -DLOD_TYPE=float2 %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library \
// RUN:   -finclude-default-header -fsyntax-only -verify=expected,dim2 \
// RUN:   -DTEXTURE=Texture2DArray -DLOD_TYPE=float2 %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library \
// RUN:   -finclude-default-header -fsyntax-only -verify=expected,dim3 \
// RUN:   -DTEXTURE=TextureCube -DLOD_TYPE=float3 %s

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   TEXTURE            resource type name
//   LOD_TYPE           CalculateLevelOfDetail location type
//
// Check prefixes:
//   dim2               diagnostics naming a 2-component offset or location
//                      vector
//   dim3               diagnostics naming a 3-component offset or location
//                      vector

TEXTURE<float4> tex;
SamplerState samp;

void main() {
  LOD_TYPE loc = (LOD_TYPE)0;

  tex.CalculateLevelOfDetail(samp, loc);
  tex.CalculateLevelOfDetailUnclamped(samp, loc);

  // expected-error@+2 {{too few arguments to function call, expected 2, have 1}}
  // expected-note@* {{'CalculateLevelOfDetail' declared here}}
  tex.CalculateLevelOfDetail(samp);

  // expected-error@+2 {{too few arguments to function call, expected 2, have 1}}
  // expected-note@* {{'CalculateLevelOfDetailUnclamped' declared here}}
  tex.CalculateLevelOfDetailUnclamped(samp);

  // expected-error@+2 {{too many arguments to function call, expected 2, have 3}}
  // expected-note@* {{'CalculateLevelOfDetail' declared here}}
  tex.CalculateLevelOfDetail(samp, loc, 0);

  // expected-error@+2{{too many arguments to function call, expected 2, have 3}}
  // expected-note@* {{'CalculateLevelOfDetailUnclamped' declared here}}
  tex.CalculateLevelOfDetailUnclamped(samp, loc, 0);

  // dim2-error@+2 {{cannot initialize a parameter of type 'vector<float, 2>' (vector of 2 'float' values) with an lvalue of type 'const char[8]'}}
  // dim3-error@+1 {{cannot initialize a parameter of type 'vector<float, 3>' (vector of 3 'float' values) with an lvalue of type 'const char[8]'}}
  tex.CalculateLevelOfDetail(samp, "invalid");

  // dim2-error@+2 {{cannot initialize a parameter of type 'vector<float, 2>' (vector of 2 'float' values) with an lvalue of type 'const char[8]'}}
  // dim3-error@+1 {{cannot initialize a parameter of type 'vector<float, 3>' (vector of 3 'float' values) with an lvalue of type 'const char[8]'}}
  tex.CalculateLevelOfDetailUnclamped(samp, "invalid");
}

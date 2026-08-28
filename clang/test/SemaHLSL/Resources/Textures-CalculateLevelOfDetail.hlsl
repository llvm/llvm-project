// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library \
// RUN:   -finclude-default-header -fsyntax-only -verify=expected,dim2 \
// RUN:   -DLOD_TYPE=float2 -DTEXTURE=Texture2D %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library \
// RUN:   -finclude-default-header -fsyntax-only -verify=expected,dim2 \
// RUN:   -DLOD_TYPE=float2 -DTEXTURE=Texture2DArray %s

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   LOD_TYPE           CalculateLevelOfDetail location type
//   TEXTURE            resource type name
//
// Check prefixes:
//   dim2               diagnostics naming a 2-component offset or location
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

  // dim2-error@+1 {{cannot initialize a parameter of type 'vector<float, 2>' (vector of 2 'float' values) with an lvalue of type 'const char[8]'}}
  tex.CalculateLevelOfDetail(samp, "invalid");

  // dim2-error@+1 {{cannot initialize a parameter of type 'vector<float, 2>' (vector of 2 'float' values) with an lvalue of type 'const char[8]'}}
  tex.CalculateLevelOfDetailUnclamped(samp, "invalid");
}

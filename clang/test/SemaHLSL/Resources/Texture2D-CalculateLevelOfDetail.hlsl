// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -finclude-default-header -fsyntax-only -verify %s

Texture2D<float4> tex;
SamplerState samp;

void main() {
  float2 loc = float2(0, 0);

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

  // expected-error@+1 {{cannot initialize a parameter of type 'vector<float, 2>' (vector of 2 'float' values) with an lvalue of type 'const char[8]'}}
  tex.CalculateLevelOfDetail(samp, "invalid");

  // expected-error@+1 {{cannot initialize a parameter of type 'vector<float, 2>' (vector of 2 'float' values) with an lvalue of type 'const char[8]'}}
  tex.CalculateLevelOfDetailUnclamped(samp, "invalid");
}

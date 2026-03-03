// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -finclude-default-header -fsyntax-only -verify %s

Texture2D<float4> tex;
SamplerState samp;

void main() {
  float2 loc = float2(0, 0);
  float lod = 0;
  int2 offset = int2(0, 0);

  tex.SampleLevel(samp, loc, lod);
  tex.SampleLevel(samp, loc, lod, offset);

  // Too few arguments.
  tex.SampleLevel(samp, loc); // expected-error {{no matching member function for call to 'SampleLevel'}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 2 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 2 were provided}}

  // Too many arguments.
  tex.SampleLevel(samp, loc, lod, offset, 0); // expected-error {{no matching member function for call to 'SampleLevel'}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 5 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 5 were provided}}

  // Invalid argument types.
  tex.SampleLevel(samp, loc, "invalid"); // expected-error {{no matching member function for call to 'SampleLevel'}}
  // expected-note@*:* {{no known conversion from 'const char[8]' to 'float' for 3rd argument}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 3 were provided}}
}

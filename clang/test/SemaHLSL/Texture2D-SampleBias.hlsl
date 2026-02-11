// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -finclude-default-header -fsyntax-only -verify %s

Texture2D<float4> tex;
SamplerState samp;

void main() {
  float2 loc = float2(0, 0);
  float bias = 0;
  int2 offset = int2(0, 0);
  float clamp = 0;

  tex.SampleBias(samp, loc, bias);
  tex.SampleBias(samp, loc, bias, offset);
  tex.SampleBias(samp, loc, bias, offset, clamp);

  // Too few arguments.
  tex.SampleBias(samp, loc); // expected-error {{no matching member function for call to 'SampleBias'}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 2 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 2 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 5 arguments, but 2 were provided}}

  // Too many arguments.
  tex.SampleBias(samp, loc, bias, offset, clamp, 0); // expected-error {{no matching member function for call to 'SampleBias'}}
  // expected-note@*:* {{candidate function not viable: requires 5 arguments, but 6 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 6 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 6 were provided}}

  // Invalid argument types.
  tex.SampleBias(samp, loc, bias, offset, "invalid"); // expected-error {{no matching member function for call to 'SampleBias'}}
  // expected-note@*:* {{no known conversion from 'const char[8]' to 'float' for 5th argument}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 5 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 5 were provided}}
}
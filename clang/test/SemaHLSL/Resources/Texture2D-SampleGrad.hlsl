// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -finclude-default-header -fsyntax-only -verify %s

Texture2D<float4> tex;
SamplerState samp;

void main() {
  float2 loc = float2(0, 0);
  float2 ddx = float2(0, 0);
  float2 ddy = float2(0, 0);
  int2 offset = int2(0, 0);
  float clamp = 0;

  tex.SampleGrad(samp, loc, ddx, ddy);
  tex.SampleGrad(samp, loc, ddx, ddy, offset);
  tex.SampleGrad(samp, loc, ddx, ddy, offset, clamp);

  // Too few arguments.
  tex.SampleGrad(samp, loc, ddx); // expected-error {{no matching member function for call to 'SampleGrad'}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 3 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 5 arguments, but 3 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 6 arguments, but 3 were provided}}

  // Too many arguments.
  tex.SampleGrad(samp, loc, ddx, ddy, offset, clamp, 0); // expected-error {{no matching member function for call to 'SampleGrad'}}
  // expected-note@*:* {{candidate function not viable: requires 6 arguments, but 7 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 5 arguments, but 7 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 7 were provided}}

  // Invalid argument types.
  tex.SampleGrad(samp, loc, ddx, ddy, offset, "invalid"); // expected-error {{no matching member function for call to 'SampleGrad'}}
  // expected-note@*:* {{no known conversion from 'const char[8]' to 'float' for 6th argument}}
  // expected-note@*:* {{candidate function not viable: requires 5 arguments, but 6 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 6 were provided}}
}

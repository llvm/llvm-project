// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -fsyntax-only -finclude-default-header -verify %s

Texture2D<float4> t;
SamplerState s;

void main(float2 loc) {
  t.Sample(s, loc);
  t.Sample(s, loc, int2(1, 2));
  
  // expected-error@+4 {{no matching member function for call to 'Sample'}}
  // expected-note@*:* {{candidate function not viable: requires 2 arguments, but 1 was provided}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 1 was provided}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 1 was provided}}
  t.Sample(loc);

  t.Sample(s, loc, int2(1, 2), 1.0);

  // expected-error@+4 {{no matching member function for call to 'Sample'}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 5 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 5 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 2 arguments, but 5 were provided}}
  t.Sample(s, loc, int2(1, 2), 1.0, 1.0);

  // expected-error@+4 {{no matching member function for call to 'Sample'}}
  // expected-note@*:* {{candidate function not viable: no known conversion from 'SamplerState' to 'vector<int, 2>' (vector of 2 'int' values) for 3rd argument}}
  // expected-note@*:* {{candidate function not viable: requires 2 arguments, but 3 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 3 were provided}}
  t.Sample(s, loc, s);

  // expected-error@+4 {{no matching member function for call to 'Sample'}}
  // expected-note@*:* {{candidate function not viable: no known conversion from 'SamplerState' to 'float' for 4th argument}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 4 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 2 arguments, but 4 were provided}}
  t.Sample(s, loc, int2(1, 2), s);

  // Test with wrong coordinate dimension.
  // Note: float implicitly converts to float2 (splat), so no error here.
  t.Sample(s, loc.x);

  // Test with wrong offset dimension.
  // Note: int implicitly converts to int2 (splat), so no error here.
  t.Sample(s, loc, 1);
}

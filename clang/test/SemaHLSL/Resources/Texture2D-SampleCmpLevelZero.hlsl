// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -fsyntax-only -finclude-default-header -verify %s

// expected-error@* {{'SampleCmp' and 'SampleCmpLevelZero' require resource to contain a floating point type}}

Texture2D<float4> t;
Texture2D<int4> t_int;
SamplerComparisonState s;
SamplerState s2;

void main(float2 loc, float cmp) {
  t.SampleCmpLevelZero(s, loc, cmp);
  t.SampleCmpLevelZero(s, loc, cmp, int2(1, 2));

  // expected-note@*:* {{in instantiation of member function 'hlsl::Texture2D<vector<int, 4>>::SampleCmpLevelZero' requested here}}
  t_int.SampleCmpLevelZero(s, loc, cmp);

  // expected-error@+3 {{no matching member function for call to 'SampleCmpLevelZero'}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 1 was provided}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 1 was provided}}
  t.SampleCmpLevelZero(loc);

  // expected-error@+3 {{no matching member function for call to 'SampleCmpLevelZero'}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 5 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 5 were provided}}
  t.SampleCmpLevelZero(s, loc, cmp, int2(1, 2), 1.0f);

  // expected-error@+3 {{no matching member function for call to 'SampleCmpLevelZero'}}
  // expected-note@*:* {{candidate function not viable: no known conversion from 'SamplerState' to 'hlsl::SamplerComparisonState' for 1st argument}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 3 were provided}}
  t.SampleCmpLevelZero(s2, loc, cmp);

  // expected-error@+3 {{no matching member function for call to 'SampleCmpLevelZero'}}
  // expected-note@*:* {{candidate function not viable: no known conversion from 'SamplerComparisonState' to 'vector<int, 2>' (vector of 2 'int' values) for 4th argument}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 4 were provided}}
  t.SampleCmpLevelZero(s, loc, cmp, s);
}
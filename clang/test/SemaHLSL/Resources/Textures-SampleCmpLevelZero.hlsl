// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -fsyntax-only -finclude-default-header -verify=expected,dim2 \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DTEXTURE=Texture2D -DCOORD_TYPE=float2 %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -fsyntax-only -finclude-default-header -verify=expected,dim2 \
// RUN:   -DOFFSET_ARG="int2(1, 2)" -DTEXTURE=Texture2DArray \
// RUN:   -DCOORD_TYPE=float3 %s

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   OFFSET_ARG         a literal offset argument
//   TEXTURE            resource type name
//   COORD_TYPE         sample location type (DIM components plus the array
//                      slice)
//
// Check prefixes:
//   dim2               diagnostics naming a 2-component offset or location
//                      vector
//
// expected-error@* {{'SampleCmp' and 'SampleCmpLevelZero' require resource to contain a floating point type}}

TEXTURE<float4> t;
TEXTURE<int4> t_int;
SamplerComparisonState s;
SamplerState s2;

void main(COORD_TYPE loc, float cmp) {
  t.SampleCmpLevelZero(s, loc, cmp);
  t.SampleCmpLevelZero(s, loc, cmp, OFFSET_ARG);

  // expected-note-re@*:* {{in instantiation of member function 'hlsl::Texture{{.+}}::SampleCmpLevelZero' requested here}}
  t_int.SampleCmpLevelZero(s, loc, cmp);

  // expected-error@+3 {{no matching member function for call to 'SampleCmpLevelZero'}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 1 was provided}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 1 was provided}}
  t.SampleCmpLevelZero(loc);

  // expected-error@+3 {{no matching member function for call to 'SampleCmpLevelZero'}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 5 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 5 were provided}}
  t.SampleCmpLevelZero(s, loc, cmp, OFFSET_ARG, 1.0f);

  // expected-error@+3 {{no matching member function for call to 'SampleCmpLevelZero'}}
  // expected-note@*:* {{candidate function not viable: no known conversion from 'SamplerState' to 'hlsl::SamplerComparisonState' for 1st argument}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 3 were provided}}
  t.SampleCmpLevelZero(s2, loc, cmp);

  // expected-error@+3 {{no matching member function for call to 'SampleCmpLevelZero'}}
  // dim2-note@*:* {{candidate function not viable: no known conversion from 'SamplerComparisonState' to 'vector<int, 2>' (vector of 2 'int' values) for 4th argument}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 4 were provided}}
  t.SampleCmpLevelZero(s, loc, cmp, s);
}

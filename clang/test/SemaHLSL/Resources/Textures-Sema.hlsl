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

TEXTURE<float4> t;
SamplerState s;

void main(COORD_TYPE loc) {
  t.Sample(s, loc);
  t.Sample(s, loc, OFFSET_ARG);

  // expected-error@+4 {{no matching member function for call to 'Sample'}}
  // expected-note@*:* {{candidate function not viable: requires 2 arguments, but 1 was provided}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 1 was provided}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 1 was provided}}
  t.Sample(loc);

  t.Sample(s, loc, OFFSET_ARG, 1.0);

  // expected-error@+4 {{no matching member function for call to 'Sample'}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 5 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 5 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 2 arguments, but 5 were provided}}
  t.Sample(s, loc, OFFSET_ARG, 1.0, 1.0);

  // expected-error@+4 {{no matching member function for call to 'Sample'}}
  // dim2-note@*:* {{candidate function not viable: no known conversion from 'SamplerState' to 'vector<int, 2>' (vector of 2 'int' values) for 3rd argument}}
  // expected-note@*:* {{candidate function not viable: requires 2 arguments, but 3 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 3 were provided}}
  t.Sample(s, loc, s);

  // expected-error@+4 {{no matching member function for call to 'Sample'}}
  // expected-note@*:* {{candidate function not viable: no known conversion from 'SamplerState' to 'float' for 4th argument}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 4 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 2 arguments, but 4 were provided}}
  t.Sample(s, loc, OFFSET_ARG, s);

  // Test with wrong coordinate dimension.
  // Note: float implicitly converts to float2/float3 (splat), so no error here.
  t.Sample(s, loc.x);

  // Test with wrong offset dimension.
  // Note: int implicitly converts to int2 (splat), so no error here.
  t.Sample(s, loc, 1);
}

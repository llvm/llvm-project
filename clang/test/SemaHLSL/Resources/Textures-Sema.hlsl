// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -fsyntax-only -finclude-default-header -verify=expected,offset,dim2 \
// RUN:   -DHAS_OFFSET -DOFFSET_ARG="int2(1, 2)" -DTEXTURE=Texture2D \
// RUN:   -DCOORD_TYPE=float2 %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -fsyntax-only -finclude-default-header -verify=expected,offset,dim2 \
// RUN:   -DHAS_OFFSET -DOFFSET_ARG="int2(1, 2)" -DTEXTURE=Texture2DArray \
// RUN:   -DCOORD_TYPE=float3 %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -fsyntax-only -finclude-default-header \
// RUN:   -verify=expected,nooffset,dim3 -DOFFSET_ARG="int3(1, 2, 3)" \
// RUN:   -DTEXTURE=TextureCube -DCOORD_TYPE=float3 %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -fsyntax-only -finclude-default-header \
// RUN:   -verify=expected,nooffset,dim3 -DOFFSET_ARG="int3(1, 2, 3)" \
// RUN:   -DTEXTURE=TextureCubeArray -DCOORD_TYPE=float4 %s

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   HAS_OFFSET         defined for types whose sampling and gathering methods
//                      have overloads taking an offset
//   OFFSET_ARG         a literal offset argument
//   TEXTURE            resource type name
//   COORD_TYPE         sample location type (DIM components plus the array
//                      slice)
//
// Check prefixes:
//   offset             diagnostics for types that have offset overloads
//   dim2               diagnostics naming a 2-component offset or location
//                      vector
//   nooffset           diagnostics for types that have no offset overloads
//   dim3               diagnostics naming a 3-component offset or location
//                      vector
//
// HAS_OFFSET, OFFSET_TYPE and OFFSET_ARG as the matching preprocessor

// TextureCube only has the Sample(SamplerState, float3) overload, so calls with
// a wrong argument count are reported as a plain arity mismatch there, while the
// 2D textures report a failed overload resolution.

TEXTURE<float4> t;
SamplerState s;

void main(COORD_TYPE loc) {
  t.Sample(s, loc);

  // expected-note@*:* {{candidate function not viable: requires 2 arguments, but 1 was provided}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 1 was provided}}
  // offset-note@*:* {{candidate function not viable: requires 4 arguments, but 1 was provided}}
  // expected-error@+1 {{no matching member function for call to 'Sample'}}
  t.Sample(loc);

  // offset-note@*:* {{candidate function not viable: requires 4 arguments, but 5 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 5 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 2 arguments, but 5 were provided}}
  // expected-error@+1 {{no matching member function for call to 'Sample'}}
  t.Sample(s, loc, OFFSET_ARG, 1.0, 1.0);

  // Test with wrong coordinate dimension.
  // Note: float implicitly converts to float2/float3 (splat), so no error here.
  t.Sample(s, loc.x);

#ifdef HAS_OFFSET
  t.Sample(s, loc, OFFSET_ARG);

  t.Sample(s, loc, OFFSET_ARG, 1.0);

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

  // Test with wrong offset dimension.
  // Note: int implicitly converts to int2 (splat), so no error here.
  t.Sample(s, loc, 1);
#else
  // This type has no overload that takes an offset, but it does have one that
  // takes a clamp, so the 3rd parameter is the clamp.
  t.Sample(s, loc, 1.0);

  // Passing an offset therefore selects the clamp overload and truncates.
  // nooffset-warning@+1 {{implicit conversion turns vector to scalar}}
  t.Sample(s, loc, OFFSET_ARG);

  // nooffset-note@*:* {{candidate function not viable: requires 3 arguments, but 4 were provided}}
  // nooffset-note@*:* {{candidate function not viable: requires 2 arguments, but 4 were provided}}
  // nooffset-error@+1 {{no matching member function for call to 'Sample'}}
  t.Sample(s, loc, OFFSET_ARG, 1.0);
#endif
}

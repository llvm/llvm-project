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


TEXTURE<float4> t;
TEXTURE<int4> t_int;
SamplerComparisonState s;
SamplerState s2;

void main(COORD_TYPE loc, float cmp) {
  t.SampleCmp(s, loc, cmp);

#ifdef HAS_OFFSET
  t.SampleCmp(s, loc, cmp, OFFSET_ARG);
  t.SampleCmp(s, loc, cmp, OFFSET_ARG, 1.0f);
#else
  // This type has no overload that takes an offset, but it does have one that
  // takes a clamp, so the 4th parameter is the clamp.
  t.SampleCmp(s, loc, cmp, 1.0f);

  // Passing an offset therefore selects the clamp overload and truncates.
  // nooffset-warning@+1 {{implicit conversion turns vector to scalar}}
  t.SampleCmp(s, loc, cmp, OFFSET_ARG);
#endif

  // expected-error@* {{'SampleCmp' and 'SampleCmpLevelZero' require resource to contain a floating point type}}
  // expected-note-re@*:* {{in instantiation of member function 'hlsl::Texture{{.+}}::SampleCmp' requested here}}
  t_int.SampleCmp(s, loc, cmp);

  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 1 was provided}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 1 was provided}}
  // offset-note@*:* {{candidate function not viable: requires 5 arguments, but 1 was provided}}
  // expected-error@+1 {{no matching member function for call to 'SampleCmp'}}
  t.SampleCmp(loc);

  // offset-note@*:* {{candidate function not viable: requires 5 arguments, but 6 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 6 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 6 were provided}}
  // expected-error@+1 {{no matching member function for call to 'SampleCmp'}}
  t.SampleCmp(s, loc, cmp, OFFSET_ARG, 1.0f, 1.0f);

  // expected-note@*:* {{candidate function not viable: no known conversion from 'SamplerState' to 'hlsl::SamplerComparisonState' for 1st argument}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 3 were provided}}
  // offset-note@*:* {{candidate function not viable: requires 5 arguments, but 3 were provided}}
  // expected-error@+1 {{no matching member function for call to 'SampleCmp'}}
  t.SampleCmp(s2, loc, cmp);

#ifdef HAS_OFFSET
  // expected-error@+4 {{no matching member function for call to 'SampleCmp'}}
  // dim2-note@*:* {{candidate function not viable: no known conversion from 'SamplerComparisonState' to 'vector<int, 2>' (vector of 2 'int' values) for 4th argument}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 4 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 5 arguments, but 4 were provided}}
  t.SampleCmp(s, loc, cmp, s);

  // expected-error@+4 {{no matching member function for call to 'SampleCmp'}}
  // expected-note@*:* {{candidate function not viable: no known conversion from 'SamplerComparisonState' to 'float' for 5th argument}}
  // expected-note@*:* {{candidate function not viable: requires 4 arguments, but 5 were provided}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 5 were provided}}
  t.SampleCmp(s, loc, cmp, OFFSET_ARG, s);
#endif
}

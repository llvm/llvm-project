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

// expected-error@* {{'SampleCmp' and 'SampleCmpLevelZero' require resource to contain a floating point type}}

TEXTURE<float4> t;
TEXTURE<int4> t_int;
SamplerComparisonState s;
SamplerState s2;

void main(COORD_TYPE loc, float cmp) {
  t.SampleCmpLevelZero(s, loc, cmp);

#ifdef HAS_OFFSET
  t.SampleCmpLevelZero(s, loc, cmp, OFFSET_ARG);
#else
  // This type has no overload that takes an offset.
  // nooffset-note@* {{'SampleCmpLevelZero' declared here}}
  // nooffset-error@+1 {{too many arguments to function call, expected 3, have 4}}
  t.SampleCmpLevelZero(s, loc, cmp, OFFSET_ARG);
#endif

  // expected-note-re@*:* {{in instantiation of member function 'hlsl::Texture{{.+}}::SampleCmpLevelZero' requested here}}
  t_int.SampleCmpLevelZero(s, loc, cmp);

  // offset-note@*:* {{candidate function not viable: requires 3 arguments, but 1 was provided}}
  // offset-note@*:* {{candidate function not viable: requires 4 arguments, but 1 was provided}}
  // nooffset-note@* {{'SampleCmpLevelZero' declared here}}
  // offset-error@+2 {{no matching member function for call to 'SampleCmpLevelZero'}}
  // nooffset-error@+1 {{too few arguments to function call, expected 3, have 1}}
  t.SampleCmpLevelZero(loc);

  // offset-note@*:* {{candidate function not viable: requires 4 arguments, but 5 were provided}}
  // offset-note@*:* {{candidate function not viable: requires 3 arguments, but 5 were provided}}
  // nooffset-note@* {{'SampleCmpLevelZero' declared here}}
  // offset-error@+2 {{no matching member function for call to 'SampleCmpLevelZero'}}
  // nooffset-error@+1 {{too many arguments to function call, expected 3, have 5}}
  t.SampleCmpLevelZero(s, loc, cmp, OFFSET_ARG, 1.0f);

  // offset-note@*:* {{candidate function not viable: no known conversion from 'SamplerState' to 'hlsl::SamplerComparisonState' for 1st argument}}
  // offset-note@*:* {{candidate function not viable: requires 4 arguments, but 3 were provided}}
  // nooffset-note@*:* {{candidate constructor not viable: no known conversion from 'SamplerState' to 'const hlsl::SamplerComparisonState &' for 1st argument}}
  // offset-error@+2 {{no matching member function for call to 'SampleCmpLevelZero'}}
  // nooffset-error@+1 {{no viable conversion from 'SamplerState' to 'hlsl::SamplerComparisonState'}}
  t.SampleCmpLevelZero(s2, loc, cmp);

#ifdef HAS_OFFSET
  // expected-error@+3 {{no matching member function for call to 'SampleCmpLevelZero'}}
  // dim2-note@*:* {{candidate function not viable: no known conversion from 'SamplerComparisonState' to 'vector<int, 2>' (vector of 2 'int' values) for 4th argument}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 4 were provided}}
  t.SampleCmpLevelZero(s, loc, cmp, s);
#endif
}

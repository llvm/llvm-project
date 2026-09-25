// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -fsyntax-only -verify -finclude-default-header -DTEXTURE=Texture2D \
// RUN:   -DOFFSET_TYPE=int2 -DCOORD_TYPE=float2 %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -fsyntax-only -verify -finclude-default-header \
// RUN:   -DTEXTURE=Texture2DArray -DOFFSET_TYPE=int2 -DCOORD_TYPE=float3 %s

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   TEXTURE            resource type name
//   OFFSET_TYPE        offset type, one component per resource dimension
//   COORD_TYPE         sample location type (DIM components plus the array
//                      slice)

TEXTURE<float4> Tex;
SamplerState Samp;
SamplerComparisonState SampCmp;

void main() {
  COORD_TYPE uv = (COORD_TYPE)0.5;
  OFFSET_TYPE offset = (OFFSET_TYPE)1;
  float compare = 0.5;

  // Gather
  Tex.Gather(Samp, uv);
  Tex.Gather(Samp, uv, offset);

  // Invalid Overloads
  Tex.Gather(Samp); // expected-error {{no matching member function for call to 'Gather'}}
  Tex.Gather(Samp, uv, offset, 1); // expected-error {{no matching member function for call to 'Gather'}}

  // Gather variants
  Tex.GatherRed(Samp, uv);
  Tex.GatherGreen(Samp, uv, offset);
  Tex.GatherBlue(Samp, uv);
  Tex.GatherAlpha(Samp, uv, offset);

  // GatherCmp
  Tex.GatherCmp(SampCmp, uv, compare);
  Tex.GatherCmp(SampCmp, uv, compare, offset);

  // Invalid Overloads
  Tex.GatherCmp(SampCmp, uv); // expected-error {{no matching member function for call to 'GatherCmp'}}
  Tex.GatherCmp(SampCmp, uv, compare, offset, 1); // expected-error {{no matching member function for call to 'GatherCmp'}}

  // GatherCmp variants
  Tex.GatherCmpRed(SampCmp, uv, compare);
  Tex.GatherCmpGreen(SampCmp, uv, compare);
  Tex.GatherCmpBlue(SampCmp, uv, compare, offset);
  Tex.GatherCmpAlpha(SampCmp, uv, compare);

  // Type checks
  Tex.Gather(Samp, uv, Samp); // expected-error {{no matching member function for call to 'Gather'}}
  Tex.GatherCmp(SampCmp, uv, Samp); // expected-error {{no matching member function for call to 'GatherCmp'}}
}

// expected-note@* 0+{{candidate function not viable}}

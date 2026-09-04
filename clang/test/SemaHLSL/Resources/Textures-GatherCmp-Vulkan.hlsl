// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -fsyntax-only -verify \
// RUN:   -finclude-default-header -DTEXTURE=Texture2D -DCOORD_TYPE=float2 %s
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -fsyntax-only -verify \
// RUN:   -finclude-default-header -DTEXTURE=Texture2DArray \
// RUN:   -DCOORD_TYPE=float3 %s

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   TEXTURE            resource type name
//   COORD_TYPE         sample location type (DIM components plus the array
//                      slice)

TEXTURE<float4> Tex;
SamplerComparisonState SampCmp;

void main() {
  COORD_TYPE uv = (COORD_TYPE)0.5;
  float compare = 0.5;

  Tex.GatherCmp(SampCmp, uv, compare);
  Tex.GatherCmpRed(SampCmp, uv, compare);

  // expected-error@* {{gatherCmpGreen operations on the Vulkan target are not supported; only GatherCmp and GatherCmpRed are allowed}}
  Tex.GatherCmpGreen(SampCmp, uv, compare);

  // expected-error@* {{gatherCmpBlue operations on the Vulkan target are not supported; only GatherCmp and GatherCmpRed are allowed}}
  Tex.GatherCmpBlue(SampCmp, uv, compare);

  // expected-error@* {{gatherCmpAlpha operations on the Vulkan target are not supported; only GatherCmp and GatherCmpRed are allowed}}
  Tex.GatherCmpAlpha(SampCmp, uv, compare);
}

// expected-note@* 0+{{in instantiation of member function}}

// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -finclude-default-header -DHAS_TEXEL -DTEXTURE=RWTexture2D \
// RUN:   -DCOORD_TYPE=float2 -DGRAD_TYPE=float2 -DOFFSET_TYPE=int2 -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -finclude-default-header -DHAS_TEXEL -DTEXTURE=RWTexture2DArray \
// RUN:   -DCOORD_TYPE=float3 -DGRAD_TYPE=float2 -DOFFSET_TYPE=int2 -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -finclude-default-header -DHAS_SAMPLE -DHAS_GATHER -DHAS_LOD \
// RUN:   -DLOAD_ARG="int4(0, 0, 0, 0)" -DINDEX_ARG="uint3(0, 0, 0)" \
// RUN:   -DTEXTURE=TextureCube -DCOORD_TYPE=float3 -DOFFSET_TYPE=int3 -verify \
// RUN:   %s

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   HAS_TEXEL          defined for types that have Load and operator[]
//   LOAD_ARG           a literal Load location
//   INDEX_ARG          a literal operator[] index
//   TEXTURE            resource type name
//   COORD_TYPE         sample location type (DIM components plus the array
//                      slice)
//   GRAD_TYPE          SampleGrad ddx/ddy type, one component per resource
//                      dimension
//   OFFSET_TYPE        offset type, one component per resource dimension
//   HAS_SAMPLE         defined for types that have the Sample* methods
//   HAS_GATHER         defined for types that have the Gather* methods
//   HAS_LOD            defined for types that have CalculateLevelOfDetail*
//
// Writable (UAV) textures have no sampling, gathering or LOD methods.

TEXTURE<float4> Tex;
SamplerState Samp;
SamplerComparisonState SampCmp;

void main(COORD_TYPE uv) {
  OFFSET_TYPE offset = (OFFSET_TYPE)0;
  float compare = 0.5f;

#ifndef HAS_SAMPLE
  // expected-error-re@+1 {{no member named 'Sample' in 'hlsl::{{.*}}Texture}}
  Tex.Sample(Samp, uv);
  // expected-error-re@+1 {{no member named 'SampleLevel' in 'hlsl::{{.*}}Texture}}
  Tex.SampleLevel(Samp, uv, 0.0f);
  // expected-error-re@+1 {{no member named 'SampleBias' in 'hlsl::{{.*}}Texture}}
  Tex.SampleBias(Samp, uv, 0.0f);
  // expected-error-re@+1 {{no member named 'SampleGrad' in 'hlsl::{{.*}}Texture}}
  Tex.SampleGrad(Samp, uv, (GRAD_TYPE)0, (GRAD_TYPE)0);
  // expected-error-re@+1 {{no member named 'SampleCmp' in 'hlsl::{{.*}}Texture}}
  Tex.SampleCmp(SampCmp, uv, compare);
  // expected-error-re@+1 {{no member named 'SampleCmpLevelZero' in 'hlsl::{{.*}}Texture}}
  Tex.SampleCmpLevelZero(SampCmp, uv, compare);
#endif

#ifndef HAS_GATHER
  // expected-error-re@+1 {{no member named 'Gather' in 'hlsl::{{.*}}Texture}}
  Tex.Gather(Samp, uv);
  // expected-error-re@+1 {{no member named 'GatherRed' in 'hlsl::{{.*}}Texture}}
  Tex.GatherRed(Samp, uv);
  // expected-error-re@+1 {{no member named 'GatherGreen' in 'hlsl::{{.*}}Texture}}
  Tex.GatherGreen(Samp, uv, offset);
  // expected-error-re@+1 {{no member named 'GatherCmp' in 'hlsl::{{.*}}Texture}}
  Tex.GatherCmp(SampCmp, uv, compare);
#endif

#ifndef HAS_LOD
  // expected-error-re@+1 {{no member named 'CalculateLevelOfDetail' in 'hlsl::{{.*}}Texture}}
  (void)Tex.CalculateLevelOfDetail(Samp, uv);
  // expected-error-re@+1 {{no member named 'CalculateLevelOfDetailUnclamped' in 'hlsl::{{.*}}Texture}}
  (void)Tex.CalculateLevelOfDetailUnclamped(Samp, uv);
#endif

#ifndef HAS_TEXEL
  // expected-error-re@+1 {{no member named 'Load' in 'hlsl::{{.*}}Texture}}
  Tex.Load(LOAD_ARG);
  // expected-error@+1 {{does not provide a subscript operator}}
  (void)Tex[INDEX_ARG];
#endif
}

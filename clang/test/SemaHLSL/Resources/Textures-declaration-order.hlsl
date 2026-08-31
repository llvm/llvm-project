// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -fsyntax-only -finclude-default-header -verify -DSCALAR_FIRST %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -fsyntax-only -finclude-default-header -verify -USCALAR_FIRST %s
// RUN: %clang_cc1 -triple spirv-unknown-vulkan-library -x hlsl -fsyntax-only -finclude-default-header -verify -DSCALAR_FIRST %s
// RUN: %clang_cc1 -triple spirv-unknown-vulkan-library -x hlsl -fsyntax-only -finclude-default-header -verify -USCALAR_FIRST %s

// Texture resource classes are declared as a primary class template, used for
// scalar element types, plus a partial specialization used for vector element
// types. Both patterns are only defined on demand by HLSLExternalSemaSource, so
// completing one of them must not prevent the other one from being completed.
// See https://github.com/llvm/llvm-project/issues/212575.

// expected-no-diagnostics

#ifdef SCALAR_FIRST
Texture2D<float> Tex2D;
Texture2D<float2> Tex2DVec;
RWTexture2D<float> RWTex2D;
RWTexture2D<float2> RWTex2DVec;
Texture2DArray<float> Tex2DArray;
Texture2DArray<float2> Tex2DArrayVec;
RWTexture2DArray<float> RWTex2DArray;
RWTexture2DArray<float2> RWTex2DArrayVec;
TextureCube<float> TexCube;
TextureCube<float2> TexCubeVec;
#else
Texture2D<float2> Tex2DVec;
Texture2D<float> Tex2D;
RWTexture2D<float2> RWTex2DVec;
RWTexture2D<float> RWTex2D;
Texture2DArray<float2> Tex2DArrayVec;
Texture2DArray<float> Tex2DArray;
RWTexture2DArray<float2> RWTex2DArrayVec;
RWTexture2DArray<float> RWTex2DArray;
TextureCube<float2> TexCubeVec;
TextureCube<float> TexCube;
#endif

SamplerState Samp;

// Use members of both the primary template and the partial specialization to
// make sure both patterns really have been completed.
export void useTextures(float2 UV, float3 UVW) {
  float S = Tex2D.Sample(Samp, UV);
  float2 V = Tex2DVec.Sample(Samp, UV);
  RWTex2D[uint2(0, 0)] = S;
  RWTex2DVec[uint2(0, 0)] = V;

  float AS = Tex2DArray.Sample(Samp, UVW);
  float2 AV = Tex2DArrayVec.Sample(Samp, UVW);
  RWTex2DArray[uint3(0, 0, 0)] = AS;
  RWTex2DArrayVec[uint3(0, 0, 0)] = AV;

  float CS = TexCube.Sample(Samp, UVW);
  float2 CV = TexCubeVec.Sample(Samp, UVW);
}

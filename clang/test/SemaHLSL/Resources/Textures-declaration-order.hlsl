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
Texture1D<float> Tex1D;
Texture1D<float2> Tex1DVec;
RWTexture1D<float> RWTex1D;
RWTexture1D<float2> RWTex1DVec;
Texture1DArray<float> Tex1DArray;
Texture1DArray<float2> Tex1DArrayVec;
RWTexture1DArray<float> RWTex1DArray;
RWTexture1DArray<float2> RWTex1DArrayVec;
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
TextureCubeArray<float> TexCubeArray;
TextureCubeArray<float2> TexCubeArrayVec;
#else
Texture1D<float2> Tex1DVec;
Texture1D<float> Tex1D;
RWTexture1D<float2> RWTex1DVec;
RWTexture1D<float> RWTex1D;
Texture1DArray<float2> Tex1DArrayVec;
Texture1DArray<float> Tex1DArray;
RWTexture1DArray<float2> RWTex1DArrayVec;
RWTexture1DArray<float> RWTex1DArray;
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
TextureCubeArray<float2> TexCubeArrayVec;
TextureCubeArray<float> TexCubeArray;
#endif

SamplerState Samp;

// Use members of both the primary template and the partial specialization to
// make sure both patterns really have been completed.
export void useTextures(float U, float2 UV, float3 UVW, float4 UVWA) {
  float S1 = Tex1D.Sample(Samp, U);
  float2 V1 = Tex1DVec.Sample(Samp, U);
  RWTex1D[0] = S1;
  RWTex1DVec[0] = V1;

  float S1A = Tex1DArray.Sample(Samp, UV);
  float2 V1A = Tex1DArrayVec.Sample(Samp, UV);
  RWTex1DArray[uint2(0, 0)] = S1A;
  RWTex1DArrayVec[uint2(0, 0)] = V1A;

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

  float CAS = TexCubeArray.Sample(Samp, UVWA);
  float2 CAV = TexCubeArrayVec.Sample(Samp, UVWA);
}

// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -fsyntax-only -verify -finclude-default-header %s

Texture2D<float4> Tex;
SamplerComparisonState SampCmp;

void main() {
  float2 uv = float2(0.5, 0.5);
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

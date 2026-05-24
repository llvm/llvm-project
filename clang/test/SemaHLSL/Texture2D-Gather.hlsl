// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -fsyntax-only -verify -finclude-default-header %s

Texture2D<float4> Tex;
SamplerState Samp;
SamplerComparisonState SampCmp;

void main() {
  float2 uv = float2(0.5, 0.5);
  int2 offset = int2(1, 1);
  float compare = 0.5;

  // Gather
  // Expected: Gather(SamplerState, float2, [int2])
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
  // Expected: GatherCmp(SamplerComparisonState, float2, float, [int2])
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
  // Offset must be int2 (SamplerState fails)
  Tex.Gather(Samp, uv, Samp); // expected-error {{no matching member function for call to 'Gather'}}
  
  // Compare value must be scalar float
  Tex.GatherCmp(SampCmp, uv, Samp); // expected-error {{no matching member function for call to 'GatherCmp'}}
}

// expected-note@* 0+{{candidate function not viable}}

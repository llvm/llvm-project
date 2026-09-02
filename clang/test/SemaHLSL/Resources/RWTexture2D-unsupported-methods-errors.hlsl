// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -finclude-default-header -DTEXTURE=RWTexture2D -verify %s

// SRV-style texture methods are not available on RWTexture2D or RWTexture2DArray (UAV).

TEXTURE<float4> Tex;
SamplerState Samp;
SamplerComparisonState SampCmp;

void main(float2 uv) {
  int2 offset = int2(0, 0);
  float compare = 0.5f;

  // expected-error@+1 {{no member named 'SampleLevel' in 'hlsl::RWTexture2D}}
  Tex.SampleLevel(Samp, uv, 0.0f);
  // expected-error@+1 {{no member named 'SampleBias' in 'hlsl::RWTexture2D}}
  Tex.SampleBias(Samp, uv, 0.0f);
  // expected-error@+1 {{no member named 'SampleGrad' in 'hlsl::RWTexture2D}}
  Tex.SampleGrad(Samp, uv, float2(0), float2(0));
  // expected-error@+1 {{no member named 'SampleCmp' in 'hlsl::RWTexture2D}}
  Tex.SampleCmp(SampCmp, uv, compare);
  // expected-error@+1 {{no member named 'SampleCmpLevelZero' in 'hlsl::RWTexture2D}}
  Tex.SampleCmpLevelZero(SampCmp, uv, compare);

  // expected-error@+1 {{no member named 'Gather' in 'hlsl::RWTexture2D}}
  Tex.Gather(Samp, uv);
  // expected-error@+1 {{no member named 'GatherRed' in 'hlsl::RWTexture2D}}
  Tex.GatherRed(Samp, uv);
  // expected-error@+1 {{no member named 'GatherGreen' in 'hlsl::RWTexture2D}}
  Tex.GatherGreen(Samp, uv, offset);
  // expected-error@+1 {{no member named 'GatherCmp' in 'hlsl::RWTexture2D}}
  Tex.GatherCmp(SampCmp, uv, compare);

  // expected-error@+1 {{no member named 'CalculateLevelOfDetail' in 'hlsl::RWTexture2D}}
  (void)Tex.CalculateLevelOfDetail(Samp, uv);
  // expected-error@+1 {{no member named 'CalculateLevelOfDetailUnclamped' in 'hlsl::RWTexture2D}}
  (void)Tex.CalculateLevelOfDetailUnclamped(Samp, uv);
}

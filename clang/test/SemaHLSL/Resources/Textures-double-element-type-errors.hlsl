// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -finclude-default-header -DTEXTURE=Texture2D -DCOORD_TYPE=float2 -DLOCATION=int3 -DINDEX=uint2 -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -finclude-default-header -DTEXTURE=Texture2DArray -DCOORD_TYPE=float3 -DLOCATION=int4 -DINDEX=uint3 -verify %s

// Textures with a 'double' element type are valid declarations, but sampling
// from them and gathering on them is not supported.

TEXTURE<double2> TexVec;
TEXTURE<double> Tex;
SamplerState Samp;
SamplerComparisonState SampCmp;

void main(COORD_TYPE uv) {
  float compare = 0.5f;

  // expected-error@* {{'Sample' is not supported for resources containing 'double'}}
  // expected-note-re@*:* {{in instantiation of member function 'hlsl::Texture{{.+}}<double>::Sample' requested here}}
  Tex.Sample(Samp, uv);

  // expected-error@* {{'SampleBias' is not supported for resources containing 'double'}}
  // expected-note-re@*:* {{in instantiation of member function 'hlsl::Texture{{.+}}<double>::SampleBias' requested here}}
  Tex.SampleBias(Samp, uv, 0.5f);

  // expected-error@* {{'SampleGrad' is not supported for resources containing 'double'}}
  // expected-note-re@*:* {{in instantiation of member function 'hlsl::Texture{{.+}}<double>::SampleGrad' requested here}}
  Tex.SampleGrad(Samp, uv, float2(0, 0), float2(0, 0));

  // expected-error@* {{'SampleLevel' is not supported for resources containing 'double'}}
  // expected-note-re@*:* {{in instantiation of member function 'hlsl::Texture{{.+}}<double>::SampleLevel' requested here}}
  Tex.SampleLevel(Samp, uv, 0.0f);

  // expected-error@* {{'SampleCmp' is not supported for resources containing 'double'}}
  // expected-note-re@*:* {{in instantiation of member function 'hlsl::Texture{{.+}}<double>::SampleCmp' requested here}}
  Tex.SampleCmp(SampCmp, uv, compare);

  // expected-error@* {{'SampleCmpLevelZero' is not supported for resources containing 'double'}}
  // expected-note-re@*:* {{in instantiation of member function 'hlsl::Texture{{.+}}<double>::SampleCmpLevelZero' requested here}}
  Tex.SampleCmpLevelZero(SampCmp, uv, compare);

  // expected-error@* {{'Gather' is not supported for resources containing 'double'}}
  // expected-note-re@*:* {{in instantiation of member function 'hlsl::Texture{{.+}}<double>::Gather' requested here}}
  Tex.Gather(Samp, uv);

  // expected-error@* {{'GatherRed' is not supported for resources containing 'double'}}
  // expected-note-re@*:* {{in instantiation of member function 'hlsl::Texture{{.+}}<double>::GatherRed' requested here}}
  Tex.GatherRed(Samp, uv);

  // expected-error@* {{'GatherCmp' is not supported for resources containing 'double'}}
  // expected-note-re@*:* {{in instantiation of member function 'hlsl::Texture{{.+}}<double>::GatherCmp' requested here}}
  Tex.GatherCmp(SampCmp, uv, compare);

  // Textures containing vectors of doubles are rejected as well.
  // expected-error@* {{'Sample' is not supported for resources containing 'vector<double, 2>' (vector of 2 'double' values)}}
  // expected-note-re@*:* {{in instantiation of member function 'hlsl::Texture{{.+}}<vector<double, 2>>::Sample' requested here}}
  TexVec.Sample(Samp, uv);

  // expected-error@* {{'SampleLevel' is not supported for resources containing 'vector<double, 2>' (vector of 2 'double' values)}}
  // expected-note-re@*:* {{in instantiation of member function 'hlsl::Texture{{.+}}<vector<double, 2>>::SampleLevel' requested here}}
  TexVec.SampleLevel(Samp, uv, 0.0f);

  // expected-error@* {{'Gather' is not supported for resources containing 'vector<double, 2>' (vector of 2 'double' values)}}
  // expected-note-re@*:* {{in instantiation of member function 'hlsl::Texture{{.+}}<vector<double, 2>>::Gather' requested here}}
  TexVec.Gather(Samp, uv);

  // Loading from and subscripting textures containing doubles is allowed.
  double a = Tex.Load((LOCATION)0);
  double b = Tex[(INDEX)0];
  double2 c = TexVec.Load((LOCATION)0);
  double2 d = TexVec[(INDEX)0];
}

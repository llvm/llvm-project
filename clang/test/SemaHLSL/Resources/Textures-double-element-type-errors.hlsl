// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -finclude-default-header -DTEXTURE=Texture2D -DCOORD_TYPE=float2 \
// RUN:   -DGRAD_TYPE=float2 -DLOAD_TYPE=int3 -DINDEX_TYPE=uint2 -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -finclude-default-header -DTEXTURE=Texture2DArray \
// RUN:   -DCOORD_TYPE=float3 -DGRAD_TYPE=float2 -DLOAD_TYPE=int4 \
// RUN:   -DINDEX_TYPE=uint3 -verify %s

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   TEXTURE            resource type name
//   COORD_TYPE         sample location type (DIM components plus the array
//                      slice)
//   GRAD_TYPE          SampleGrad ddx/ddy type, one component per resource
//                      dimension
//   LOAD_TYPE          Load location type
//   INDEX_TYPE         operator[] index type
//
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
  Tex.SampleGrad(Samp, uv, (GRAD_TYPE)0, (GRAD_TYPE)0);

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
  double a = Tex.Load((LOAD_TYPE)0);
  double b = Tex[(INDEX_TYPE)0];
  double2 c = TexVec.Load((LOAD_TYPE)0);
  double2 d = TexVec[(INDEX_TYPE)0];
}

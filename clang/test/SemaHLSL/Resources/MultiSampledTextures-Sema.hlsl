// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -finclude-default-header -fsyntax-only -verify -DTEXTURE=Texture2DMS -DLOCATION_TYPE=int2 -DINDEX_TYPE=uint2 %s

// A multisampled texture supports the sample-indexed Load and operator[], but
// does not support Sample/SampleBias/SampleGrad/SampleLevel/SampleCmp, Gather,
// CalculateLevelOfDetail, or mips.

TEXTURE<float4> T;
SamplerState S;

// The sample_count template parameter is a non-type 'int' parameter, so it must
// be an integer constant expression. There is currently no HLSL-specific
// validation of the value itself (0, negative, non-power-of-2, and out-of-range
// counts are all accepted); the value is forwarded to the generated resource
// type as the multisampled sample count.
TEXTURE<float4, 4> TExplicit;       // explicit valid count
static const int SC = 8;
TEXTURE<float4, SC> TConst;         // constant-expression count

int NonConst;
TEXTURE<float4, NonConst> TBad;     // expected-error {{non-type template argument is not a constant expression}}
// expected-note@-1 {{read of non-const variable 'NonConst' is not allowed in a constant expression}}
// expected-note@-3 {{declared here}}
TEXTURE<float4, 1.5> TFloat;        // expected-error {{conversion from 'float' to 'int' is not allowed in a converted constant expression}}

// The element type has no default argument: a multisampled texture must be
// declared with an explicit element type, matching DXC. Both the bare
// template-name (shorthand) form and an empty template argument list are
// errors. (Non-multisampled textures such as Texture2D do allow these forms.)
TEXTURE TBare;                      // expected-error-re {{use of class template 'Texture2DMS{{(Array)?}}' requires template arguments}}
// expected-note@*:* {{template declaration from hidden source}}
TEXTURE<> TEmpty;                   // expected-error-re {{too few template arguments for class template 'Texture2DMS{{(Array)?}}'}}
// expected-note@*:* {{template declaration from hidden source}}

void valid() {
  // Sample-indexed Load and its offset overload.
  float4 a = T.Load((LOCATION_TYPE)0, 0);
  float4 b = T.Load((LOCATION_TYPE)0, 0, int2(1, 1));
  // operator[] reads sample 0.
  float4 c = T[(INDEX_TYPE)0];
}

void unsupported() {
  T.Sample(S, float2(0, 0));   // expected-error-re {{no member named 'Sample' in 'hlsl::Texture2DMS{{(Array)?}}<vector<float, 4>>'}}
  T.SampleLevel(S, float2(0, 0), 0); // expected-error-re {{no member named 'SampleLevel' in 'hlsl::Texture2DMS{{(Array)?}}<vector<float, 4>>'}}
  T.Gather(S, float2(0, 0));   // expected-error-re {{no member named 'Gather' in 'hlsl::Texture2DMS{{(Array)?}}<vector<float, 4>>'}}
  T.CalculateLevelOfDetail(S, float2(0, 0)); // expected-error-re {{no member named 'CalculateLevelOfDetail' in 'hlsl::Texture2DMS{{(Array)?}}<vector<float, 4>>'}}
  T.mips[0][(LOCATION_TYPE)0]; // expected-error-re {{no member named 'mips' in 'hlsl::Texture2DMS{{(Array)?}}<vector<float, 4>>'}}
}

void bad_load() {
  // Load on a multisampled texture requires a sample index.
  T.Load((LOCATION_TYPE)0);    // expected-error {{no matching member function for call to 'Load'}}
  // expected-note@*:* {{candidate function not viable: requires 2 arguments, but 1 was provided}}
  // expected-note@*:* {{candidate function not viable: requires 3 arguments, but 1 was provided}}
}

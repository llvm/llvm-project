// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -finclude-default-header -fsyntax-only -verify %s

// The `sample_count` non-type template parameter only exists on multisampled
// texture types. Every other resource type takes just an `element_type`, so
// providing a second template argument must be diagnosed as too many template
// arguments.

struct S {
  float4 F;
};

// Multisampled textures accept the sample count.
Texture2DMS<float4> MSDefaultCount;
Texture2DMS<float4, 4> MSExplicitCount;

// ...but not more than that.
Texture2DMS<float4, 4, 8> MSTooMany;
// expected-error@-1 {{too many template arguments for class template 'Texture2DMS'}}
// expected-note@*:* {{template declaration from hidden source}}

// Typed buffers.
Buffer<float4, 4> B;
// expected-error@-1 {{too many template arguments for class template 'Buffer'}}
// expected-note@*:* {{template declaration from hidden source}}
RWBuffer<float4, 4> RWB;
// expected-error@-1 {{too many template arguments for class template 'RWBuffer'}}
// expected-note@*:* {{template declaration from hidden source}}
RasterizerOrderedBuffer<float4, 4> ROB;
// expected-error@-1 {{too many template arguments for class template 'RasterizerOrderedBuffer'}}
// expected-note@*:* {{template declaration from hidden source}}

// Structured buffers.
StructuredBuffer<S, 4> SB;
// expected-error@-1 {{too many template arguments for class template 'StructuredBuffer'}}
// expected-note@*:* {{template declaration from hidden source}}
RWStructuredBuffer<S, 4> RWSB;
// expected-error@-1 {{too many template arguments for class template 'RWStructuredBuffer'}}
// expected-note@*:* {{template declaration from hidden source}}
AppendStructuredBuffer<S, 4> ASB;
// expected-error@-1 {{too many template arguments for class template 'AppendStructuredBuffer'}}
// expected-note@*:* {{template declaration from hidden source}}
ConsumeStructuredBuffer<S, 4> CSB;
// expected-error@-1 {{too many template arguments for class template 'ConsumeStructuredBuffer'}}
// expected-note@*:* {{template declaration from hidden source}}
RasterizerOrderedStructuredBuffer<S, 4> ROSB;
// expected-error@-1 {{too many template arguments for class template 'RasterizerOrderedStructuredBuffer'}}
// expected-note@*:* {{template declaration from hidden source}}

// Constant buffers.
ConstantBuffer<S, 4> CB;
// expected-error@-1 {{too many template arguments for class template 'ConstantBuffer'}}
// expected-note@*:* {{template declaration from hidden source}}

// Non-multisampled textures.
Texture2D<float4, 4> T2D;
// expected-error@-1 {{too many template arguments for class template 'Texture2D'}}
// expected-note@*:* {{template declaration from hidden source}}
RWTexture2D<float4, 4> RWT2D;
// expected-error@-1 {{too many template arguments for class template 'RWTexture2D'}}
// expected-note@*:* {{template declaration from hidden source}}
Texture2DArray<float4, 4> T2DA;
// expected-error@-1 {{too many template arguments for class template 'Texture2DArray'}}
// expected-note@*:* {{template declaration from hidden source}}
RWTexture2DArray<float4, 4> RWT2DA;
// expected-error@-1 {{too many template arguments for class template 'RWTexture2DArray'}}
// expected-note@*:* {{template declaration from hidden source}}

// Resource types that are not templates at all cannot take a sample count
// either.
ByteAddressBuffer<4> BAB;      // expected-error {{expected unqualified-id}}
RWByteAddressBuffer<4> RWBAB;  // expected-error {{expected unqualified-id}}
RasterizerOrderedByteAddressBuffer<4> ROBAB; // expected-error {{expected unqualified-id}}
SamplerState<4> Samp;          // expected-error {{expected unqualified-id}}
SamplerComparisonState<4> SampCmp; // expected-error {{expected unqualified-id}}

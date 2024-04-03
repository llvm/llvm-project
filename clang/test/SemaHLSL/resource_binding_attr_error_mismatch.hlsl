// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - -fsyntax-only %s -verify


// expected-error@+1 {{invalid register name prefix 'b' for register type 'RWBuffer' (expected 'u')}}
RWBuffer<int> a : register(b2, space1);

// expected-error@+1 {{invalid register name prefix 't' for register type 'RWBuffer' (expected 'u')}}
RWBuffer<int> b : register(t2, space1);

// NOT YET IMPLEMENTED : {{invalid register name prefix 'u' for register type 'Texture1D' (expected 't')}}
// NOT YET IMPLEMENTED Texture1D<float> tex : register(u3);

// NOT YET IMPLEMENTED : {{invalid register name prefix 's' for register type 'Texture2D' (expected 't')}}
// NOT YET IMPLEMENTED Texture2D<float> Texture : register(s0);

// NOT YET IMPLEMENTED : {{invalid register name prefix 'u' for register type 'Texture2DMS' (expected 't')}}
// NOT YET IMPLEMENTED Texture2DMS<float4, 4> T2DMS_t2 : register(u2)

// NOT YET IMPLEMENTED : {{invalid register name prefix 't' for register type 'RWTexture3D' (expected 'u')}}
// NOT YET IMPLEMENTED RWTexture3D<float4> RWT3D_u1 : register(t1)

// NOT YET IMPLEMENTED : {{invalid register name prefix 'b' for register type 'Texture2DMS' (expected 't' or 's')}}
// NOT YET IMPLEMENTED TextureCube TCube_b2 : register(B2);

// NOT YET IMPLEMENTED : {{invalid register name prefix 'b' for register type 'Texture2DMS' (expected 't')}}
// NOT YET IMPLEMENTED TextureCubeArray TCubeArray_t2 : register(b2);

// NOT YET IMPLEMENTED : {{invalid register name prefix 'b' for register type 'Texture2DMS' (expected 't' or 's')}}
// NOT YET IMPLEMENTED Texture1DArray T1DArray_t2 : register(b2);

// NOT YET IMPLEMENTED : {{invalid register name prefix 'u' for register type 'Texture2DMS' (expected 't' or 's')}}
// NOT YET IMPLEMENTED Texture2DArray T2DArray_b2 : register(B2);

// NOT YET IMPLEMENTED : {{invalid register name prefix 'u' for register type 'Texture2DMS' (expected 'b' or 'c' or 'i')}}
// NOT YET IMPLEMENTED Texture2DMSArray<float4> msTextureArray : register(t2, space2);

// NOT YET IMPLEMENTED : {{invalid register name prefix 'u' for register type 'TCubeArray_f2' (expected 't' or 's')}}
// NOT YET IMPLEMENTED TextureCubeArray TCubeArray_f2 : register(u2);

// NOT YET IMPLEMENTED : {{invalid register name prefix 'u' for register type 'TypedBuffer' (expected 't')}}
// NOT YET IMPLEMENTED TypedBuffer tbuf : register(u2);

// NOT YET IMPLEMENTED : {{invalid register name prefix 'u' for register type 'RawBuffer' (expected 't')}}
// NOT YET IMPLEMENTED RawBuffer rbuf : register(u2);

// NOT YET IMPLEMENTED : {{invalid register name prefix 't' for register type 'StructuredBuffer' (expected 'u')}}
// NOT YET IMPLEMENTED StructuredBuffer ROVStructuredBuff_t2  : register(T2);

// expected-error@+1 {{invalid register name prefix 's' for register type 'cbuffer' (expected 'b')}}
cbuffer f : register(s2, space1) {}

// NOT YET IMPLEMENTED : {{invalid register name prefix 't' for register type 'Sampler' (expected 's')}}
// Can this type just be Sampler instead of SamplerState?
// NOT YET IMPLEMENTED SamplerState MySampler : register(t3, space1);

// expected-error@+1 {{invalid register name prefix 's' for register type 'tbuffer' (expected 'b')}}
tbuffer f : register(s2, space1) {}

// NOT YET IMPLEMENTED : RTAccelerationStructure doesn't have any example tests in DXC

// NOT YET IMPLEMENTED : {{invalid register name prefix 'u' for register type 'FeedbackTexture2D' (expected 'b' or 'c' or 'i')}}
// NOT YET IMPLEMENTED FeedbackTexture2D<float> FBTex2D[3][] : register(u0, space26);

// NOT YET IMPLEMENTED : {{invalid register name prefix 'u' for register type 'FeedbackTexture2DArray' (expected 'b' or 'c' or 'i')}}
// NOT YET IMPLEMENTED FeedbackTexture2DArray<float> FBTex2DArr[3][2][] : register(u0, space27);

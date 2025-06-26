// RUN: %clang_dxc -triple dxil-pc-shadermodel6.3-library -x hlsl -o - %s -verify

#define ROOT_SIGNATURE \
    "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), " \
    "CBV(b0, visibility=SHADER_VISIBILITY_ALL), " \
    "DescriptorTable(SRV(t0, numDescriptors=3), visibility=SHADER_VISIBILITY_PIXEL), " \
    "DescriptorTable(Sampler(s0, numDescriptors=2), visibility=SHADER_VISIBILITY_PIXEL), " \
    "DescriptorTable(UAV(u0, numDescriptors=1), visibility=SHADER_VISIBILITY_ALL)"

cbuffer CB : register(b3, space2) {
  float a;
}

StructuredBuffer<int> In : register(t0);
RWStructuredBuffer<int> Out : register(u0);

RWBuffer<float> UAV : register(u3);

RWBuffer<float> UAV1 : register(u2), UAV2 : register(u4);

RWBuffer<float> UAV3 : register(space5);

float f : register(c5);

int4 intv : register(c2);

double dar[5] :  register(c3);

struct S {
  int a;
};

S s : register(c10);

// Compute Shader for UAV testing
[numthreads(8, 8, 1)]
[RootSignature(ROOT_SIGNATURE)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    In[0] = id;
    Out[0] = In[0];
}

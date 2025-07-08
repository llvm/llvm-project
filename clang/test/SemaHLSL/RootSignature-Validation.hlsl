// RUN: %clang_dxc -T cs_6_6 -E CSMain %s 2>&1 

// expected-no-diagnostics


#define ROOT_SIGNATURE \
    "CBV(b3, space=1, visibility=SHADER_VISIBILITY_ALL), " \
    "DescriptorTable(SRV(t0, space=0, numDescriptors=1), visibility=SHADER_VISIBILITY_ALL), " \
    "DescriptorTable(Sampler(s0, numDescriptors=2), visibility=SHADER_VISIBILITY_VERTEX), " \
    "DescriptorTable(UAV(u0, numDescriptors=unbounded), visibility=SHADER_VISIBILITY_ALL)"

cbuffer CB : register(b3, space1) {
  float a;
}

StructuredBuffer<int> In : register(t0, space0);
RWStructuredBuffer<int> Out : register(u0);

RWStructuredBuffer<float> UAV : register(u4294967294);

RWStructuredBuffer<float> UAV1 : register(u2), UAV2 : register(u4);

RWStructuredBuffer<float> UAV3 : register(space0);



// Compute Shader for UAV testing
[numthreads(8, 8, 1)]
[RootSignature(ROOT_SIGNATURE)]
void CSMain(uint id : SV_GroupID)
{
    Out[0] = a + id + In[0] + UAV[0] + UAV1[0] + UAV3[0];
}

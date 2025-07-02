
#define ROOT_SIGNATURE \
    "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), " \
    "CBV(b3, space=1, visibility=SHADER_VISIBILITY_ALL), " \
    "DescriptorTable(SRV(t0, space=0, numDescriptors=1), visibility=SHADER_VISIBILITY_ALL), " \
    "DescriptorTable(Sampler(s0, numDescriptors=2), visibility=SHADER_VISIBILITY_ALL), " \
    "DescriptorTable(UAV(u0, numDescriptors=unbounded), visibility=SHADER_VISIBILITY_ALL)"

cbuffer CB : register(b3, space2) {
  float a;
}

StructuredBuffer<int> In : register(t0, space0);
RWStructuredBuffer<int> Out : register(u0);

RWBuffer<float> UAV : register(u3);

RWBuffer<float> UAV1 : register(u2), UAV2 : register(u4);

RWBuffer<float> UAV3 : register(space0);



// Compute Shader for UAV testing
[numthreads(8, 8, 1)]
[RootSignature(ROOT_SIGNATURE)]
void CSMain(uint id : SV_GroupID)
{
    Out[0] = a + id + In[0] + UAV[0] + UAV1[0] + UAV3[0];
}

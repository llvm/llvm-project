// RUN: not %clang_dxc -T cs_6_6 -E CSMain %s 2>&1 | FileCheck %s

// CHECK: error: register srv (space=0, register=0) is bound to a texture or typed buffer.

RWStructuredBuffer<int> Out : register(u0);
Buffer<float> B : register(t0);
// Compute Shader for UAV testing
[numthreads(8, 8, 1)]
[RootSignature("SRV(t0), UAV(u0)")]
void CSMain(uint id : SV_GroupID)
{
    Out[0] = B[0];
}

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-library %s -emit-llvm -o - -verify

RWBuffer<uint> In : register(u0);
RWStructuredBuffer<uint> Out0 : register(u1);
RWStructuredBuffer<uint> Out1 : register(u2);

cbuffer C {
    bool Cond;
};

static RWStructuredBuffer<uint> StaticOut;

export void static_conditional_assignment(uint Idx) {
// expected-warning@+1 {{assignment of 'Cond ? Out0 : Out1' to local resource 'StaticOut' is not to the same unique global resource}}
    StaticOut = Cond ? Out0 : Out1;
    StaticOut[Idx] = In[Idx];
}

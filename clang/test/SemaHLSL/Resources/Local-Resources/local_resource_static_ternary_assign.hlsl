// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -finclude-default-header -verify %s

// Test that assigning a ternary expression to a static resource variable
// triggers a -Whlsl-explicit-binding warning.
// DXC: Error (codegen) — "non const static global resource use is disallowed in library exports"
//       and "local resource not guaranteed to map to unique global resource"

RWBuffer<uint> In : register(u0);
RWStructuredBuffer<uint> Out0 : register(u1);
RWStructuredBuffer<uint> Out1 : register(u2);

cbuffer c {
    bool cond;
};

static RWStructuredBuffer<uint> StaticOut;

void static_conditional_assignment(uint idx) {
    // expected-warning@+1 {{assignment of 'cond ? Out0 : Out1' to local resource 'StaticOut' is not to the same unique global resource}}
    StaticOut = cond ? Out0 : Out1;
    StaticOut[idx] = In[idx];
}

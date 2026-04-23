// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -finclude-default-header -verify %s

// Test that reassigning a local resource to a different global inside a
// conditional branch triggers a -Whlsl-explicit-binding warning.
// DXC: Error (codegen) — "local resource not guaranteed to map to unique global resource"

RWBuffer<uint> In : register(u0);
RWStructuredBuffer<uint> Out0 : register(u1);
RWStructuredBuffer<uint> Out1 : register(u2);

cbuffer c {
    bool cond;
};

void branched_assignment(uint idx) {
    RWStructuredBuffer<uint> Out = Out0; // expected-note {{variable 'Out' is declared here}}
    if (cond) {
        // expected-warning@+1 {{assignment of 'Out1' to local resource 'Out' is not to the same unique global resource}}
        Out = Out1;
    }
    Out[idx] = In[idx];
}

// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -finclude-default-header -verify %s

// Test that reassigning a local resource from an unbounded array element
// inside a conditional branch triggers a -Whlsl-explicit-binding warning.
// DXC: Error (codegen) — "local resource not guaranteed to map to unique global resource"

RWBuffer<uint> In : register(u0);
RWStructuredBuffer<uint> Out0 : register(u1);
RWStructuredBuffer<uint> OutArr[];

cbuffer c {
    bool cond;
};

void branched_assignment_with_array(uint idx) {
    RWStructuredBuffer<uint> Out = Out0; // expected-note {{variable 'Out' is declared here}}
    if (cond) {
        // expected-warning@+1 {{assignment of 'OutArr[0]' to local resource 'Out' is not to the same unique global resource}}
        Out = OutArr[0];
    }
    Out[idx] = In[idx];
}

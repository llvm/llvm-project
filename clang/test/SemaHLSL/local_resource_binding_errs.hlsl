// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -verify %s

RWBuffer<int> In : register(u0);
RWStructuredBuffer<int> Out0 : register(u1);
RWStructuredBuffer<int> Out1 : register(u2);
RWStructuredBuffer<int> OutArr[];

cbuffer c {
    bool cond;
};

void conditional_initialization(int idx) {
    // expected-error@+1 {{assignment to local resource 'cond ? Out0 : Out1' is not to the same unique global resource}}
    RWStructuredBuffer<int> Out = cond ? Out0 : Out1;
    Out[idx] = In[idx];
}

void branched_assignment(int idx) {
    RWStructuredBuffer<int> Out = Out0; // expected-note {{variable 'Out' is declared here}}
    if (cond) {
        // expected-error@+1 {{assignment to local resource 'Out1' is not to the same unique global resource}}
        Out = Out1;
    }
    Out[idx] = In[idx];
}

void branched_assignment_with_array(int idx) {
    RWStructuredBuffer<int> Out = Out0; // expected-note {{variable 'Out' is declared here}}
    if (cond) {
        // expected-error@+1 {{assignment to local resource 'OutArr[0]' is not to the same unique global resource}}
        Out = OutArr[0];
    }
    Out[idx] = In[idx];
}

void conditional_assignment(int idx) {
    RWStructuredBuffer<int> Out;
    // expected-error@+1 {{assignment to local resource 'cond ? Out0 : Out1' is not to the same unique global resource}}
    Out = cond ? Out0 : Out1;
    Out[idx] = In[idx];
}

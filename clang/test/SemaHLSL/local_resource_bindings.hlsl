// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -finclude-default-header -verify %s

// expected-no-diagnostics

RWBuffer<uint> In : register(u0);
RWStructuredBuffer<uint> Out0 : register(u1);
RWStructuredBuffer<uint> Out1 : register(u2);
RWStructuredBuffer<uint> OutArr[];

cbuffer c {
    bool cond;
};

void no_initial_assignment(uint idx) {
    RWStructuredBuffer<uint> Out;
    if (cond) {
        Out = Out1;
    }
    Out[idx] = In[idx];
}

void assignment_to_uninitialized(uint idx) {
    RWStructuredBuffer<uint> Out;
    Out = Out;
    Out[idx] = In[idx];
}

void same_assignment(uint idx) {
    RWStructuredBuffer<uint> Out = Out1;
    if (cond) {
        Out = Out1;
    }
    Out[idx] = In[idx];
}

void conditional_initialization_with_index(uint idx) {
    RWStructuredBuffer<uint> Out = cond ? OutArr[0] : OutArr[1];
    Out[idx] = In[idx];
}

void conditional_assignment_with_index(uint idx) {
    RWStructuredBuffer<uint> Out;
	if (cond) {
		Out = OutArr[0];
	} else {
		Out = OutArr[1];
	}
    Out[idx] = In[idx];
}

void reassignment(uint idx) {
    RWStructuredBuffer<uint> Out = Out0;
	if (cond) {
		Out = Out0;
	}
	Out[idx] = In[idx];
}

void conditional_result_in_same(uint idx) {
    RWStructuredBuffer<uint> Out = cond ? Out0 : Out0;
	Out[idx] = In[idx];
}

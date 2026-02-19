// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -verify %s

// expected-no-diagnostics

RWBuffer<int> In : register(u0);
RWStructuredBuffer<int> Out0 : register(u1);
RWStructuredBuffer<int> Out1 : register(u2);
RWStructuredBuffer<int> OutArr[];

cbuffer c {
    bool cond;
};

void no_initial_assignment(int idx) {
    RWStructuredBuffer<int> Out;
    if (cond) {
        Out = Out1;
    }
    Out[idx] = In[idx];
}

void assignment_to_uninitialized(int idx) {
    RWStructuredBuffer<int> Out;
    Out = Out;
    Out[idx] = In[idx];
}

void same_assignment(int idx) {
    RWStructuredBuffer<int> Out = Out1;
    if (cond) {
        Out = Out1;
    }
    Out[idx] = In[idx];
}

void conditional_initialization_with_index(int idx) {
    RWStructuredBuffer<int> Out = cond ? OutArr[0] : OutArr[1];
    Out[idx] = In[idx];
}

void conditional_assignment_with_index(int idx) {
    RWStructuredBuffer<int> Out;
	if (cond) {
		Out = OutArr[0];
	} else {
		Out = OutArr[1];
	}
    Out[idx] = In[idx];
}

void reassignment(int idx) {
    RWStructuredBuffer<int> Out = Out0;
	if (cond) {
		Out = Out0;
	}
	Out[idx] = In[idx];
}

void conditional_result_in_same(int idx) {
    RWStructuredBuffer<int> Out = cond ? Out0 : Out0;
	Out[idx] = In[idx];
}

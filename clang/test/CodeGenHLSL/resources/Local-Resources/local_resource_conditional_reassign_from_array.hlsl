// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWBuffer<uint> In : register(u0);
RWStructuredBuffer<uint> Out0 : register(u1);
RWStructuredBuffer<uint> OutArr[];

cbuffer C {
    bool Cond;
};

void branched_assignment_with_array(uint Idx) {
    RWStructuredBuffer<uint> Out = Out0;
    if (Cond) {
        Out = OutArr[0];
    }
    Out[Idx] = In[Idx];
}

[numthreads(64, 1, 1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    branched_assignment_with_array(Tid.x);
}

// Binding wrapper for In (register(u0, space0)) is emitted.
// CHECK-DAG: call {{.*}}__createFromBinding{{.*}}@_ZL2In,
// Binding wrapper for Out0 (register(u1, space0)) is emitted.
// CHECK-DAG: call {{.*}}__createFromBinding{{.*}}@_ZL4Out0,

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -O1 \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWBuffer<uint> In : register(u0);
RWStructuredBuffer<uint> Out0 : register(u1);
RWStructuredBuffer<uint> Out1 : register(u2);

cbuffer C {
    bool Cond;
};

void branched_assignment(uint Idx) {
    RWStructuredBuffer<uint> Out = Out0;
    if (Cond) {
        Out = Out1;
    }
    Out[Idx] = In[Idx];
}

[numthreads(64, 1, 1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    branched_assignment(Tid.x);
}

// CHECK-LABEL: define {{.*}}@main(
// Binding for In (register(u0, space0)) is emitted.
// CHECK-DAG: call {{.*}}handlefrombinding{{.*}}(i32 0, i32 0,
// Binding for Out0 (register(u1, space0)) is emitted.
// CHECK-DAG: call {{.*}}handlefrombinding{{.*}}(i32 0, i32 1,
// Binding for Out1 (register(u2, space0)) is emitted.
// CHECK-DAG: call {{.*}}handlefrombinding{{.*}}(i32 0, i32 2,

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -O1 \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWByteAddressBuffer G0 : register(u0);
RWByteAddressBuffer G1 : register(u1);

RWByteAddressBuffer Pick(bool C)
{
    if (C) return G0;
    return G1;
}

[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID)
{
    RWByteAddressBuffer Buf = Pick(Tid.x > 0);
    Buf.Store(Tid.x * 4, 42);
}

// CHECK-LABEL: define {{.*}}@main(
// Binding for G0 (register(u0, space0)) is emitted.
// CHECK-DAG: call {{.*}}handlefrombinding{{.*}}(i32 0, i32 0,
// Binding for G1 (register(u1, space0)) is emitted.
// CHECK-DAG: call {{.*}}handlefrombinding{{.*}}(i32 0, i32 1,

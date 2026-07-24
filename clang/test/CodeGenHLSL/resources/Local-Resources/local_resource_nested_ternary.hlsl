// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -O1 \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWByteAddressBuffer GBuf0 : register(u0);
RWByteAddressBuffer GBuf1 : register(u1);

uint Pass_IfAlias(bool Cond, uint Idx) {
    RWByteAddressBuffer Buf;
    Buf = Cond ? GBuf0 : GBuf1;
    Buf.Store(Idx * 4, 14);

    return 14;
}

[numthreads(8,8,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    uint Idx = Tid.x + Tid.y * 8;
    Pass_IfAlias(Idx < 32, Idx);
}

// CHECK-LABEL: define {{.*}}@main(
// Binding for GBuf0 (register(u0, space0)) is emitted.
// CHECK-DAG: call {{.*}}handlefrombinding{{.*}}(i32 0, i32 0,
// Binding for GBuf1 (register(u1, space0)) is emitted.
// CHECK-DAG: call {{.*}}handlefrombinding{{.*}}(i32 0, i32 1,

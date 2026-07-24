// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWByteAddressBuffer GBuf0 : register(u0);
RWByteAddressBuffer GBuf1 : register(u1);
RWByteAddressBuffer GBuf2 : register(u2);

uint Pass_DeepPhi(bool A, bool B, uint Idx) {
    RWByteAddressBuffer Buf;

    if (A)
        Buf = B ? GBuf0 : GBuf1;
    else
        Buf = GBuf2;

    Buf.Store(Idx * 4, 25);

    return 25;
}

[numthreads(8,8,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    uint Idx = Tid.x + Tid.y * 8;
    Pass_DeepPhi(true, false, Idx);
}

// Binding wrapper for GBuf0 (register(u0, space0)) is emitted.
// CHECK-DAG: call {{.*}}__createFromBinding{{.*}}@_ZL5GBuf0,
// Binding wrapper for GBuf1 (register(u1, space0)) is emitted.
// CHECK-DAG: call {{.*}}__createFromBinding{{.*}}@_ZL5GBuf1,
// Binding wrapper for GBuf2 (register(u2, space0)) is emitted.
// CHECK-DAG: call {{.*}}__createFromBinding{{.*}}@_ZL5GBuf2,

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWByteAddressBuffer GBuf0 : register(u0);
RWByteAddressBuffer GBuf1 : register(u1);

RWByteAddressBuffer Pass_MultipleReturns(bool Cond, uint Idx) {
    RWByteAddressBuffer Buf = GBuf0;
    if (Cond) {
        Buf.Store(Idx * 4, 1);
        return Buf;
    }
    Buf = GBuf1;
    Buf.Store(Idx * 4, 2);
    return Buf;
}

[numthreads(8,8,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    uint Idx = Tid.x + Tid.y * 8;
    RWByteAddressBuffer Result = Pass_MultipleReturns(Idx < 32, Idx);
    Result.Store(64 * 4 + Idx * 4, Idx);
}

// Binding wrapper for GBuf0 (register(u0, space0)) is emitted.
// CHECK-DAG: call {{.*}}__createFromBinding{{.*}}@_ZL5GBuf0,
// Binding wrapper for GBuf1 (register(u1, space0)) is emitted.
// CHECK-DAG: call {{.*}}__createFromBinding{{.*}}@_ZL5GBuf1,

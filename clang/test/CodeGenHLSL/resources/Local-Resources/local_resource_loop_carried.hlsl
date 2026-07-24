// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWByteAddressBuffer GBuf0 : register(u0);
RWByteAddressBuffer GBufArray[4] : register(u10);

uint Pass_LoopCarried(int Iterations, uint Idx) {
    RWByteAddressBuffer Buf = GBuf0;

    for (int I=0;I<Iterations;I++)
        Buf = GBufArray[I & 3];

    Buf.Store(Idx * 4, 26);

    return 26;
}

[numthreads(8,8,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    uint Idx = Tid.x + Tid.y * 8;
    Pass_LoopCarried(15, Idx);
}

// Binding wrapper for GBuf0 (register(u0, space0)) is emitted.
// CHECK-DAG: call {{.*}}__createFromBinding{{.*}}@_ZL5GBuf0,

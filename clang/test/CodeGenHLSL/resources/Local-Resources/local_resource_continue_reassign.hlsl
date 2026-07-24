// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWByteAddressBuffer GBuf0 : register(u0);
RWByteAddressBuffer GBuf1 : register(u1);

[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    RWByteAddressBuffer Buf = GBuf0;

    for (uint I = 0; I < 4; I++) {
        if (I == 2) {
            Buf = GBuf1;
            continue;
        }
        Buf.Store(I * 4, I);
    }

    Buf.Store(Tid.x * 4, 99);
}

// Binding wrapper for GBuf0 (register(u0, space0)) is emitted.
// CHECK-DAG: call {{.*}}__createFromBinding{{.*}}@_ZL5GBuf0,
// Binding wrapper for GBuf1 (register(u1, space0)) is emitted.
// CHECK-DAG: call {{.*}}__createFromBinding{{.*}}@_ZL5GBuf1,

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWByteAddressBuffer GBuf0 : register(u0);

uint Pass_StaticLocal(uint Idx) {
    static RWByteAddressBuffer Buf = GBuf0;
    Buf.Store(Idx * 4, 1);

    return 1;
}

[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    Pass_StaticLocal(Tid.x);
}

// Binding wrapper for GBuf0 (register(u0, space0)) is emitted.
// CHECK-DAG: call {{.*}}__createFromBinding{{.*}}@_ZL5GBuf0,

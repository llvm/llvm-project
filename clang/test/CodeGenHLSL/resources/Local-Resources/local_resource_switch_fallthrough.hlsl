// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -O1 \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWByteAddressBuffer GBuf0 : register(u0);
RWByteAddressBuffer GBuf1 : register(u1);
RWByteAddressBuffer GBuf2 : register(u2);

uint Pass_SwitchFallthrough(int V, uint Idx) {
    RWByteAddressBuffer Buf = GBuf0;

    switch (V) {
        case 0: Buf = GBuf1;
        case 1: Buf = GBuf2; break;
    }

    Buf.Store(Idx * 4, 30);

    return 30;
}

[numthreads(8,8,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    uint Idx = Tid.x + Tid.y * 8;
    Pass_SwitchFallthrough(0, Idx);
}

// CHECK-LABEL: define {{.*}}@main(
// Binding for GBuf0 (register(u0, space0)) is emitted.
// CHECK-DAG: call {{.*}}handlefrombinding{{.*}}(i32 0, i32 0,
// Binding for GBuf2 (register(u2, space0)) is emitted.
// CHECK-DAG: call {{.*}}handlefrombinding{{.*}}(i32 0, i32 2,
// Local resource resolves unambiguously; GBuf1's binding is folded away.
// CHECK-NOT: handlefrombinding{{.*}}(i32 0, i32 1,

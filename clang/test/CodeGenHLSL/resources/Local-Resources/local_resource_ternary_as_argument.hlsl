// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -O1 \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWByteAddressBuffer GBuf0 : register(u0);
RWByteAddressBuffer GBuf1 : register(u1);

void Helper(RWByteAddressBuffer Buf, uint Offset, uint Value) {
    Buf.Store(Offset, Value);
}

[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    bool Cond = Tid.x > 0;
    Helper(Cond ? GBuf0 : GBuf1, Tid.x * 4, 42);
}

// CHECK-LABEL: define {{.*}}@main(
// Binding for GBuf0 (register(u0, space0)) is emitted.
// CHECK-DAG: call {{.*}}handlefrombinding{{.*}}(i32 0, i32 0,
// Binding for GBuf1 (register(u1, space0)) is emitted.
// CHECK-DAG: call {{.*}}handlefrombinding{{.*}}(i32 0, i32 1,

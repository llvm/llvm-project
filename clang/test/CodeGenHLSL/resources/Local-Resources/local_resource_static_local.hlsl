// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWByteAddressBuffer GBuf0 : register(u0);

void Pass_StaticLocal(uint Idx) {
    static RWByteAddressBuffer Buf = GBuf0;
    Buf.Store(Idx * 4, 1);
}

[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    Pass_StaticLocal(Tid.x);
}

// Binding wrapper for GBuf0 (register(u0, space0)) is emitted.
// The static local Buf is copy-constructed from GBuf0; its Store call writes through the static.
// CHECK-DAG: call {{.*}}__createFromBinding{{.*}}@_ZL5GBuf0,
// CHECK-LABEL: define {{.*}}@_Z16Pass_StaticLocal
// CHECK: call void @{{.*}}RWByteAddressBufferC1{{.*}}(ptr {{.*}} @_ZZ16Pass_StaticLocaljE3Buf, ptr {{.*}} @_ZL5GBuf0
// CHECK: call void @{{.*}}RWByteAddressBuffer5Store{{.*}}(ptr {{.*}} @_ZZ16Pass_StaticLocaljE3Buf,

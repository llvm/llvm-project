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

// Buf is initialized from GBuf0 and reassigned to GBuf1 in the continue-branch.
// Verify the actual bindings that Buf carries, not just that the global wrappers exist.
// CHECK-LABEL: define {{.*}}@_Z4mainDv3_j(
// CHECK: %Buf = alloca %"class.hlsl::RWByteAddressBuffer"
// Init: Buf is copy-constructed from GBuf0.
// CHECK: call void @{{.*}}RWByteAddressBufferC1{{.*}}(ptr {{.*}} %Buf, ptr {{.*}} @_ZL5GBuf0
// Reassign in the continue-branch: Buf is copy-assigned from GBuf1.
// CHECK: call {{.*}}ptr @{{.*}}RWByteAddressBufferaS{{.*}}(ptr {{.*}} %Buf, ptr {{.*}} @_ZL5GBuf1
// Both Stores go through Buf (never through a global directly).
// CHECK: call void @{{.*}}RWByteAddressBuffer5Store{{.*}}(ptr {{.*}} %Buf,
// CHECK: call void @{{.*}}RWByteAddressBuffer5Store{{.*}}(ptr {{.*}} %Buf,

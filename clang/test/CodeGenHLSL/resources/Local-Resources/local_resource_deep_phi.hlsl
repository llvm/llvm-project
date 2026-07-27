// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWByteAddressBuffer GBuf0 : register(u0);
RWByteAddressBuffer GBuf1 : register(u1);
RWByteAddressBuffer GBuf2 : register(u2);

void Pass_DeepPhi(bool A, bool B, uint Idx) {
    RWByteAddressBuffer Buf;

    if (A)
        Buf = B ? GBuf0 : GBuf1;
    else
        Buf = GBuf2;

    Buf.Store(Idx * 4, 25);
}

[numthreads(8,8,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    uint Idx = Tid.x + Tid.y * 8;
    Pass_DeepPhi(true, false, Idx);
}

// Verify Buf's actual bindings (not just that global wrappers exist):
//   * true branch: ternary produces a phi(GBuf0, GBuf1) that gets copy-assigned into Buf
//   * false branch: GBuf2 gets copy-assigned into Buf
//   * the Store goes through Buf.
// CHECK-LABEL: define {{.*}}@_Z12Pass_DeepPhi
// CHECK: %Buf = alloca %"class.hlsl::RWByteAddressBuffer"
// Ternary's phi selects between GBuf0 (B==true) and GBuf1 (B==false).
// CHECK: %{{.*}} = phi ptr {{.*}}@_ZL5GBuf0{{.*}}@_ZL5GBuf1
// The phi's result is copy-assigned into Buf.
// CHECK: call {{.*}}ptr @{{.*}}RWByteAddressBufferaS{{.*}}(ptr {{.*}} %Buf, ptr {{.*}} %{{[A-Za-z_][A-Za-z0-9_.]*}})
// The else branch copy-assigns GBuf2 into Buf.
// CHECK: call {{.*}}ptr @{{.*}}RWByteAddressBufferaS{{.*}}(ptr {{.*}} %Buf, ptr {{.*}} @_ZL5GBuf2
// Store goes through Buf.
// CHECK: call void @{{.*}}RWByteAddressBuffer5Store{{.*}}(ptr {{.*}} %Buf,

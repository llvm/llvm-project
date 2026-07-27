// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWByteAddressBuffer Out : register(u0);
RWByteAddressBuffer Aux : register(u1);

[numthreads(1,1,1)]
void main() {
    RWByteAddressBuffer Src[2];
    Src[0] = Out;
    Src[1] = Aux;
    RWByteAddressBuffer Dst[2] = Src;
    Dst[0].Store(0, 42);
}

// Verify the actual binding chain: Src[0]<-Out, Src[1]<-Aux, and Dst[0].Store
// goes through a %Dst-derived pointer (so Dst[0] carries Out's binding via the array copy).
// CHECK-LABEL: define {{.*}}@_Z4mainv(
// CHECK: %Src = alloca [2 x %"class.hlsl::RWByteAddressBuffer"]
// CHECK: %Dst = alloca [2 x %"class.hlsl::RWByteAddressBuffer"]
// After the default-init arrayctor loop, Src[0] is copy-assigned from Out and Src[1] from Aux.
// CHECK: arrayctor.cont:
// CHECK: %[[SRC0:[^ ]+]] = getelementptr inbounds {{.*}}ptr %Src, i32 0, i32 0
// CHECK-NEXT: %{{.*}} = call {{.*}}RWByteAddressBufferaS{{.*}}(ptr {{.*}} %[[SRC0]], ptr {{.*}} @_ZL3Out
// CHECK: %[[SRC1:[^ ]+]] = getelementptr inbounds {{.*}}ptr %Src, i32 0, i32 1
// CHECK-NEXT: %{{.*}} = call {{.*}}RWByteAddressBufferaS{{.*}}(ptr {{.*}} %[[SRC1]], ptr {{.*}} @_ZL3Aux
// After the elementwise Dst = Src copy loop, the Store must go through Dst[0].
// CHECK: arrayinit.end:
// CHECK: %[[DST0:[^ ]+]] = getelementptr inbounds {{.*}}ptr %Dst, i32 0, i32 0
// CHECK-NEXT: call void @{{.*}}RWByteAddressBuffer5Store{{.*}}(ptr {{.*}} %[[DST0]],

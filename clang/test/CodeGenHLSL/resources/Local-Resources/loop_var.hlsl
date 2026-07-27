// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWByteAddressBuffer GBuf : register(u0);

[numthreads(1,1,1)]
void main() {
    for (RWByteAddressBuffer Buf = GBuf; true; ) {
        Buf.Store(0, 42);
        break;
    }
}

// Binding wrapper for GBuf (register(u0, space0)) is emitted.
// The for-loop's local Buf is copy-constructed from GBuf; the Store call in the loop body writes through %Buf.
// CHECK-DAG: call {{.*}}__createFromBinding{{.*}}@_ZL4GBuf,
// CHECK-LABEL: define {{.*}}@_Z4mainv(
// CHECK: %Buf = alloca %"class.hlsl::RWByteAddressBuffer"
// CHECK: call void @{{.*}}RWByteAddressBufferC1{{.*}}(ptr {{.*}} %Buf, ptr {{.*}} @_ZL4GBuf
// CHECK: for.body:
// CHECK: call void @{{.*}}RWByteAddressBuffer5Store{{.*}}(ptr {{.*}} %Buf,

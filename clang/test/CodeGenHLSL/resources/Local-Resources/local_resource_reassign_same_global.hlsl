// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -O1 \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWByteAddressBuffer In : register(u0);
RWByteAddressBuffer Out0 : register(u1);

[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    RWByteAddressBuffer Out = Out0;
    if (Tid.x == 0) {
        Out = Out0;
    }
    Out.Store(0, In.Load(0));
}

// CHECK-LABEL: define {{.*}}@main(
// In (u0, space0) is loaded from and Out0 (u1, space0) is stored to; Out only ever holds Out0.
// CHECK: %[[HI:[^ ]+]] = tail call {{.*}}handlefrombinding{{.*}}(i32 0, i32 0,
// CHECK: %[[HO:[^ ]+]] = tail call {{.*}}handlefrombinding{{.*}}(i32 0, i32 1,
// In.Load(0) reads via In's handle.
// CHECK: %[[PI:[^ ]+]] = call ptr {{.*}}getpointer{{.*}}(target({{.*}}) %[[HI]], i32 0)
// CHECK: %[[V:[^ ]+]] = load i32, ptr %[[PI]]
// Out.Store(0, In.Load(0)) writes via Out0's handle.
// CHECK: %[[PO:[^ ]+]] = call ptr {{.*}}getpointer{{.*}}(target({{.*}}) %[[HO]], i32 0)
// CHECK: store i32 %[[V]], ptr %[[PO]]

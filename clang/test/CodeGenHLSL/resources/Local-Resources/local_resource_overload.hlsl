// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -O1 \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWByteAddressBuffer GBuf : register(u0);
RWStructuredBuffer<uint> GSB : register(u1);

void DoStore(RWByteAddressBuffer Buf) {
    Buf.Store(0, 42);
}
void DoStore(RWStructuredBuffer<uint> Buf) {
    Buf[0] = 77;
}

[numthreads(1,1,1)]
void main() {
    RWByteAddressBuffer LocalBuf = GBuf;
    RWStructuredBuffer<uint> LocalSB = GSB;
    DoStore(LocalBuf);
    DoStore(LocalSB);
}

// CHECK-LABEL: define {{.*}}@main(
// LocalBuf is bound to GBuf (u0, space0) and Stores 42; LocalSB is bound to GSB (u1, space0) and Stores 77.
// CHECK: %[[HB:[^ ]+]] = tail call {{.*}}handlefrombinding{{.*}}(i32 0, i32 0,
// CHECK: %[[HS:[^ ]+]] = tail call {{.*}}handlefrombinding{{.*}}(i32 0, i32 1,
// CHECK: %[[PB:[^ ]+]] = call {{.*}}ptr {{.*}}getpointer{{.*}}(target({{.*}}) %[[HB]], i32 0)
// CHECK: store i32 42, ptr %[[PB]]
// CHECK: %[[PS:[^ ]+]] = call {{.*}}ptr {{.*}}getpointer{{.*}}(target({{.*}}) %[[HS]], i32 0)
// CHECK: store i32 77, ptr %[[PS]]

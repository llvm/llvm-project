// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -O1 \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWByteAddressBuffer GBuf0 : register(u0);
RWByteAddressBuffer GBuf1 : register(u1);

[numthreads(1,1,1)]
void main() {
    RWByteAddressBuffer Buf = GBuf0;
    {
        // Inner scope shadow: Buf refers to GBuf1 here.
        RWByteAddressBuffer Buf = GBuf1;
        Buf.Store(0, 42);
    }
    // Outer scope: after the inner block exits, Buf refers to GBuf0 again.
    Buf.Store(4, 99);
}

// CHECK-LABEL: define {{.*}}@main(
// Inner-scope Buf resolves to GBuf1 (u1) — Store(0, 42) writes through GBuf1's handle.
// Outer-scope Buf (after inner block) resolves to GBuf0 (u0) — Store(4, 99) writes through GBuf0's handle.
// CHECK: %[[H0:[^ ]+]] = tail call {{.*}}handlefrombinding{{.*}}(i32 0, i32 0,
// CHECK: %[[H1:[^ ]+]] = tail call {{.*}}handlefrombinding{{.*}}(i32 0, i32 1,
// CHECK: %[[PIN:[^ ]+]] = call ptr {{.*}}getpointer{{.*}}(target({{.*}}) %[[H1]], i32 0)
// CHECK: store i32 42, ptr %[[PIN]]
// CHECK: %[[POUT:[^ ]+]] = call ptr {{.*}}getpointer{{.*}}(target({{.*}}) %[[H0]], i32 4)
// CHECK: store i32 99, ptr %[[POUT]]

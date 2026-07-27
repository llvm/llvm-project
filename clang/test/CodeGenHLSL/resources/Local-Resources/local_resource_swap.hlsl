// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -O1 \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWByteAddressBuffer GBuf0 : register(u0);
RWByteAddressBuffer GBuf1 : register(u1);

[numthreads(1,1,1)]
void main() {
    RWByteAddressBuffer A = GBuf0;
    RWByteAddressBuffer B = GBuf1;
    RWByteAddressBuffer Temp = A;
    A = B;
    B = Temp;
    A.Store(0, 1);
    B.Store(0, 2);
}

// CHECK-LABEL: define {{.*}}@main(
// After swap, A holds GBuf1 (u1, space0) and B holds GBuf0 (u0, space0).
// CHECK: %[[H0:[^ ]+]] = tail call {{.*}}handlefrombinding{{.*}}(i32 0, i32 0,
// CHECK: %[[H1:[^ ]+]] = tail call {{.*}}handlefrombinding{{.*}}(i32 0, i32 1,
// A.Store(0, 1) writes through GBuf1's handle.
// CHECK: %[[PA:[^ ]+]] = call ptr {{.*}}getpointer{{.*}}(target({{.*}}) %[[H1]], i32 0)
// CHECK: store i32 1, ptr %[[PA]]
// B.Store(0, 2) writes through GBuf0's handle.
// CHECK: %[[PB:[^ ]+]] = call ptr {{.*}}getpointer{{.*}}(target({{.*}}) %[[H0]], i32 0)
// CHECK: store i32 2, ptr %[[PB]]

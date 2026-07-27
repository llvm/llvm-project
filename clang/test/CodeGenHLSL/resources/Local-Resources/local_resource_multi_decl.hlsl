// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -O1 \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWByteAddressBuffer GBuf0 : register(u0);
RWByteAddressBuffer GBuf1 : register(u1);

[numthreads(1,1,1)]
void main() {
    RWByteAddressBuffer A = GBuf0, B = GBuf1;
    A.Store(0, 42);
    B.Store(0, 42);
}

// CHECK-LABEL: define {{.*}}@main(
// A holds GBuf0 (u0, space0) and B holds GBuf1 (u1, space0), each stored to at offset 0 with value 42.
// CHECK: %[[H0:[^ ]+]] = tail call {{.*}}handlefrombinding{{.*}}(i32 0, i32 0,
// CHECK: %[[H1:[^ ]+]] = tail call {{.*}}handlefrombinding{{.*}}(i32 0, i32 1,
// CHECK: %[[P0:[^ ]+]] = call ptr {{.*}}getpointer{{.*}}(target({{.*}}) %[[H0]], i32 0)
// CHECK: store i32 42, ptr %[[P0]]
// CHECK: %[[P1:[^ ]+]] = call ptr {{.*}}getpointer{{.*}}(target({{.*}}) %[[H1]], i32 0)
// CHECK: store i32 42, ptr %[[P1]]

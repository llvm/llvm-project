// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -O1 \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWByteAddressBuffer Out : register(u0);
RWByteAddressBuffer Aux : register(u1);

[numthreads(1,1,1)]
void main() {
    RWByteAddressBuffer Arr[2];
    RWByteAddressBuffer Arr2[1];
    Arr[0] = Out;
    Arr2[0] = Aux;
    Arr[0].Store(0, 42);
    Arr2[0].Store(4, 99);
}

// CHECK-LABEL: define {{.*}}@main(
// Out's handle (u0, space0) flows into Store 42 at offset 0; Aux's (u1, space0) into Store 99 at offset 4.
// CHECK: %[[H0:[^ ]+]] = tail call {{.*}}handlefrombinding{{.*}}(i32 0, i32 0,
// CHECK: %[[H1:[^ ]+]] = tail call {{.*}}handlefrombinding{{.*}}(i32 0, i32 1,
// CHECK: %[[P0:[^ ]+]] = call ptr {{.*}}getpointer{{.*}}(target({{.*}}) %[[H0]], i32 0)
// CHECK: store i32 42, ptr %[[P0]]
// CHECK: %[[P1:[^ ]+]] = call ptr {{.*}}getpointer{{.*}}(target({{.*}}) %[[H1]], i32 4)
// CHECK: store i32 99, ptr %[[P1]]

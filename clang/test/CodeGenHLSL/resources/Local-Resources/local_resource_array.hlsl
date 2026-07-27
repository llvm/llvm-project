// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -O1 \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWByteAddressBuffer Out : register(u0);
RWByteAddressBuffer Aux : register(u1);

[numthreads(1,1,1)]
void main() {
    RWByteAddressBuffer Arr[2];
    Arr[0] = Out;
    Arr[1] = Aux;
    Arr[0].Store(0, 42);
    Arr[1].Store(4, 99);
}

// CHECK-LABEL: define {{.*}}@main(
// Binding for Out (register(u0, space0)) is emitted.
// CHECK-DAG: call {{.*}}handlefrombinding{{.*}}(i32 0, i32 0,
// Binding for Aux (register(u1, space0)) is emitted.
// CHECK-DAG: call {{.*}}handlefrombinding{{.*}}(i32 0, i32 1,

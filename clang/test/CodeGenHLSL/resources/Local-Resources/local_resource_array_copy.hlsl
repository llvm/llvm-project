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

// Binding wrapper for Out (register(u0, space0)) is emitted.
// CHECK-DAG: call {{.*}}__createFromBinding{{.*}}@_ZL3Out,
// Binding wrapper for Aux (register(u1, space0)) is emitted.
// CHECK-DAG: call {{.*}}__createFromBinding{{.*}}@_ZL3Aux,

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
// Binding for GBuf0 (register(u0, space0)) is emitted.
// CHECK-DAG: call {{.*}}handlefrombinding{{.*}}(i32 0, i32 0,
// Binding for GBuf1 (register(u1, space0)) is emitted.
// CHECK-DAG: call {{.*}}handlefrombinding{{.*}}(i32 0, i32 1,

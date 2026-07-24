// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -O1 \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWByteAddressBuffer GBufArray[4] : register(u0);

[numthreads(1,1,1)]
void main() {
    RWByteAddressBuffer Buf;
    Buf = GBufArray[2];
    Buf = GBufArray[3];
    Buf.Store(0, 42);
}

// CHECK-LABEL: define {{.*}}@main(
// Binding for space0_u0 (register(u0, space0)) is emitted.
// CHECK-DAG: call {{.*}}handlefrombinding{{.*}}(i32 0, i32 0,

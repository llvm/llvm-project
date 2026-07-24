// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -O1 \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWByteAddressBuffer GBuf : register(u0);

RWByteAddressBuffer GetLocal() {
    RWByteAddressBuffer Buf = GBuf;
    return Buf;
}

[numthreads(1,1,1)]
void main() {
    RWByteAddressBuffer Tmp = GetLocal();
    Tmp.Store(0, 42);
}

// CHECK-LABEL: define {{.*}}@main(
// Binding for GBuf (register(u0, space0)) is emitted.
// CHECK-DAG: call {{.*}}handlefrombinding{{.*}}(i32 0, i32 0,

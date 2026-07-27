// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -O1 \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWByteAddressBuffer GBuf : register(u0);

void Level2(RWByteAddressBuffer Buf, uint Idx) {
    Buf.Store(Idx, 42);
}
void Level1(RWByteAddressBuffer Buf, uint Idx) {
    Level2(Buf, Idx);
}

[numthreads(1,1,1)]
void main() {
    RWByteAddressBuffer Local = GBuf;
    Level1(Local, 0);
    Level1(Local, 4);
}

// CHECK-LABEL: define {{.*}}@main(
// Binding for GBuf (register(u0, space0)) is emitted.
// CHECK-DAG: call {{.*}}handlefrombinding{{.*}}(i32 0, i32 0,

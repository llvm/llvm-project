// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWByteAddressBuffer GBuf : register(u0);

[numthreads(1,1,1)]
void main() {
    for (RWByteAddressBuffer Buf = GBuf; true; ) {
        Buf.Store(0, 42);
        break;
    }
}

// Binding wrapper for GBuf (register(u0, space0)) is emitted.
// CHECK-DAG: call {{.*}}__createFromBinding{{.*}}@_ZL4GBuf,

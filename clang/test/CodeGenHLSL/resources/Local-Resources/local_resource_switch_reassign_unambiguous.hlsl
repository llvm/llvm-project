// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -O1 \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWByteAddressBuffer GBuf0 : register(u0);
RWByteAddressBuffer GBuf1 : register(u1);
RWByteAddressBuffer GBuf2 : register(u2);

[numthreads(1,1,1)]
void main() {
    RWByteAddressBuffer Buf = GBuf0;
    switch (1) {
        case 1: Buf = GBuf1; break;
        case 2: Buf = GBuf2; break;
    }
    Buf.Store(0, 42);
}

// CHECK-LABEL: define {{.*}}@main(
// Binding for GBuf1 (register(u1, space0)) is emitted.
// CHECK-DAG: call {{.*}}handlefrombinding{{.*}}(i32 0, i32 1,
// Local resource resolves unambiguously; GBuf0's binding is folded away.
// CHECK-NOT: handlefrombinding{{.*}}(i32 0, i32 0,
// Local resource resolves unambiguously; GBuf2's binding is folded away.
// CHECK-NOT: handlefrombinding{{.*}}(i32 0, i32 2,

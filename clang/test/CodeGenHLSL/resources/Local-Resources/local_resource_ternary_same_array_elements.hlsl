// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -O1 \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWByteAddressBuffer In : register(u0);
RWByteAddressBuffer OutArr[] : register(u1);

[numthreads(1,1,1)]
void main() {
    RWByteAddressBuffer Out = true ? OutArr[0] : OutArr[1];
    Out.Store(0, In.Load(0));
}

// CHECK-LABEL: define {{.*}}@main(
// Binding for In (register(u0, space0)) is emitted.
// CHECK-DAG: call {{.*}}handlefrombinding{{.*}}(i32 0, i32 0,
// Binding for space0_u1 (register(u1, space0)) is emitted.
// CHECK-DAG: call {{.*}}handlefrombinding{{.*}}(i32 0, i32 1,

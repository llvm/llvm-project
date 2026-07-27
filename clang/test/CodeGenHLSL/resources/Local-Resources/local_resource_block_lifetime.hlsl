// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -O1 \
// RUN:   -Wno-hlsl-explicit-binding -o - | FileCheck %s

RWByteAddressBuffer Out : register(u0);

[numthreads(1,1,1)]
void main() {
    RWByteAddressBuffer Buf;
    {
        Buf = Out;
    }
    Buf.Store(0, 42);
}

// CHECK-LABEL: define {{.*}}@main(
// Out's handle materializes at (u0, space0), then flows into the Store of 42 at offset 0.
// CHECK: %[[H:[^ ]+]] = tail call {{.*}}handlefrombinding{{.*}}(i32 0, i32 0,
// CHECK: %[[P:[^ ]+]] = call ptr {{.*}}getpointer{{.*}}(target({{.*}}) %[[H]], i32 0)
// CHECK: store i32 42, ptr %[[P]]

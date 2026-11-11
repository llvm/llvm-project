// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.6-compute %s -emit-llvm -O1 -o - | FileCheck %s

RWByteAddressBuffer GBuf0 : register(u0);
RWByteAddressBuffer GBuf1 : register(u1);

[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    RWByteAddressBuffer A = GBuf0;
    RWByteAddressBuffer B = GBuf1;
    (true ? A : B) = GBuf0;
    A.Store(Tid.x * 4, 1);
    B.Store(Tid.x * 4, 2);
}

// CHECK-LABEL: define {{.*}}@main(
// After (true ? A : B) = GBuf0, A stays bound to GBuf0 (u0) and B stays bound to GBuf1 (u1).
// CHECK: %[[H0:[^ ]+]] = tail call {{.*}}handlefrombinding{{.*}}(i32 0, i32 0,
// CHECK: %[[H1:[^ ]+]] = tail call {{.*}}handlefrombinding{{.*}}(i32 0, i32 1,
// A.Store(Tid.x*4, 1) writes through GBuf0.
// CHECK: %[[PA:[^ ]+]] = call ptr {{.*}}getpointer{{.*}}(target({{.*}}) %[[H0]], i32 %{{[^,)]+}})
// CHECK: store i32 1, ptr %[[PA]]
// B.Store(Tid.x*4, 2) writes through GBuf1.
// CHECK: %[[PB:[^ ]+]] = call ptr {{.*}}getpointer{{.*}}(target({{.*}}) %[[H1]], i32 %{{[^,)]+}})
// CHECK: store i32 2, ptr %[[PB]]

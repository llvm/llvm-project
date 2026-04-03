// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - 2>&1 | llvm-cxxfilt | FileCheck %s

// Test that a nested ternary resource selection produces a warning in
// clang but still generates valid IR.
//
// DXC: passes sema but fails codegen with two errors:
//   "local resource not guaranteed to map to unique global resource."
//   (one for each ternary level)

RWByteAddressBuffer gBuf0 : register(u0);
RWByteAddressBuffer gBuf1 : register(u1);
RWByteAddressBuffer gBuf2 : register(u2);

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    bool c1 = tid.x > 0;
    bool c2 = tid.y > 0;
    RWByteAddressBuffer buf = c1 ? gBuf0 : (c2 ? gBuf1 : gBuf2);
    buf.Store(tid.x * 4, 42);
}

// CHECK: warning: assignment of 'c1 ? gBuf0 : (c2 ? gBuf1 : gBuf2)' to local resource 'buf' is not to the same unique global resource
// CHECK: define {{.*}} @main(
// CHECK-NOT: error:

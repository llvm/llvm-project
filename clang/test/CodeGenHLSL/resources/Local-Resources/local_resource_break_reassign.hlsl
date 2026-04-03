// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - 2>&1 | llvm-cxxfilt | FileCheck %s

// Test that resource reassignment before a break in a loop triggers a
// warning in clang but still produces valid IR.
//
// DXC: passes sema but fails codegen with:
//   "local resource not guaranteed to map to unique global resource."

RWByteAddressBuffer gBuf0 : register(u0);
RWByteAddressBuffer gBuf1 : register(u1);

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    RWByteAddressBuffer buf = gBuf0;
    for (uint i = 0; i < 4; i++) {
        if (i == tid.x) break;
        buf = gBuf1;
    }
    buf.Store(tid.x * 4, 42);
}

// CHECK: warning: assignment of 'gBuf1' to local resource 'buf' is not to the same unique global resource
// CHECK: define {{.*}} @main(
// CHECK-NOT: error:

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - 2>&1 | llvm-cxxfilt | FileCheck %s

// Test that a ternary resource expression passed directly as a function
// argument produces valid IR in clang.
//
// DXC: passes sema but fails codegen with:
//   "local resource not guaranteed to map to unique global resource."

RWByteAddressBuffer gBuf0 : register(u0);
RWByteAddressBuffer gBuf1 : register(u1);

void Helper(RWByteAddressBuffer buf, uint offset, uint value) {
    buf.Store(offset, value);
}

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    bool cond = tid.x > 0;
    Helper(cond ? gBuf0 : gBuf1, tid.x * 4, 42);
}

// CHECK-NOT: error:
// CHECK: define {{.*}} @main(

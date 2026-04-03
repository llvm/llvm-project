// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - 2>&1 | llvm-cxxfilt | FileCheck %s

// Test that a wave-conditional resource reassignment produces a warning
// in clang but still generates valid IR.
//
// DXC: passes sema but fails codegen with:
//   "local resource not guaranteed to map to unique global resource."

RWByteAddressBuffer gBuf0 : register(u0);
RWByteAddressBuffer gBuf1 : register(u1);

uint Fail_WaveUniform(uint offset, uint value) {
    RWByteAddressBuffer buf = gBuf0;
    if (WaveActiveAllTrue(true))
        buf = gBuf1;
    // expected-warning: assignment of 'gBuf1' to local resource 'buf' is not to the same unique global resource
    buf.Store(offset, value);

    return value;
}

// CHECK: warning: assignment of 'gBuf1' to local resource 'buf' is not to the same unique global resource
// CHECK: define {{.*}} @Fail_WaveUniform(
// CHECK: define {{.*}} @main(
// CHECK-NOT: error:

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    Fail_WaveUniform(tid.x * 4, 10);
}

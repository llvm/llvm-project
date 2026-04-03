// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// Test that a local resource array with only some elements initialized
// (others left default) compiles successfully.
//
// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBuf0 : register(u0);

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    RWByteAddressBuffer arr[4];
    arr[0] = gBuf0;
    arr[1] = gBuf0;
    arr[tid.x & 3].Store(tid.x * 4, 42);
}

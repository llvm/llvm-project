// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// Test that a local resource array can be dynamically indexed at runtime.
//
// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBufArray[4] : register(u0);

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    RWByteAddressBuffer localArr[2];
    localArr[0] = gBufArray[0];
    localArr[1] = gBufArray[1];
    localArr[tid.x & 1].Store(tid.x * 4, 42);
}

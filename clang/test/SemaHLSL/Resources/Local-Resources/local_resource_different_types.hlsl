// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// Test that two different resource types can coexist as local variables
// in the same function.
//
// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBuf0 : register(u0);
RWStructuredBuffer<uint> gSB : register(u1);

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    RWByteAddressBuffer localBuf = gBuf0;
    RWStructuredBuffer<uint> localSB = gSB;
    localBuf.Store(tid.x * 4, 42);
    localSB[tid.x] = 99;
}

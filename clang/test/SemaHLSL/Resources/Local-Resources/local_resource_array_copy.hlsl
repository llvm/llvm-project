// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// Test that a local resource array can be copied to another local array.
//
// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBufArray[4] : register(u0);

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    RWByteAddressBuffer src[2];
    src[0] = gBufArray[0];
    src[1] = gBufArray[1];
    RWByteAddressBuffer dst[2] = src;
    dst[0].Store(tid.x * 4, 42);
}

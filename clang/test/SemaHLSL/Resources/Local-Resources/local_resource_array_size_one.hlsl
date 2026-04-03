// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// Test that a local resource array of size 1 works correctly. This is
// an edge case where the compiler might apply different optimizations
// compared to larger arrays.
//
// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBuf0 : register(u0);

uint Pass_ArraySizeOne(uint idx) {
    RWByteAddressBuffer arr[1];
    arr[0] = gBuf0;
    arr[0].Store(idx * 4, 42);
    return 42;
}

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    Pass_ArraySizeOne(tid.x);
}

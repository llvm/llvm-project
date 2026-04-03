// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// Test that two resource variables can be declared in a single
// declaration statement. This exercises the C++ multi-declarator
// path with resource types.
//
// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBuf0 : register(u0);
RWByteAddressBuffer gBuf1 : register(u1);

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    RWByteAddressBuffer a = gBuf0, b = gBuf1;
    a.Store(tid.x * 4, 1);
    b.Store(tid.x * 4, 2);
}

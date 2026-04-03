// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// Test that swapping two local resource variables through a temporary
// is accepted without warnings. This exercises multi-variable
// reassignment that does not change which global each variable maps to.
//
// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBuf0 : register(u0);
RWByteAddressBuffer gBuf1 : register(u1);

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    RWByteAddressBuffer a = gBuf0;
    RWByteAddressBuffer b = gBuf1;
    RWByteAddressBuffer temp = a;
    a = b;
    b = temp;
    a.Store(tid.x * 4, 1);
    b.Store(tid.x * 4, 2);
}

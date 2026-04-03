// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// Test that a ternary expression can be used as an lvalue for resource
// assignment. This exercises lvalue analysis of the ternary operator
// with resource types.
//
// DXC: ICEs with "Internal compiler error: LLVM Assert" (even at sema).

RWByteAddressBuffer gBuf0 : register(u0);
RWByteAddressBuffer gBuf1 : register(u1);

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    RWByteAddressBuffer a = gBuf0;
    RWByteAddressBuffer b = gBuf1;
    bool cond = tid.x > 0;
    (cond ? a : b) = gBuf0;
    a.Store(tid.x * 4, 1);
    b.Store(tid.x * 4, 2);
}

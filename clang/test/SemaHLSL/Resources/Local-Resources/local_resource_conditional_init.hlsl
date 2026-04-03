// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// Test that a resource declared without initialization and only assigned
// in one branch produces valid IR in clang.
//
// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBuf0 : register(u0);

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    RWByteAddressBuffer buf;
    if (tid.x > 0)
        buf = gBuf0;
    buf.Store(tid.x * 4, 42);
}

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// Test that self-assignment of a local resource compiles cleanly.
//
// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBuf0 : register(u0);

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    RWByteAddressBuffer buf = gBuf0;
    buf = buf;
    buf.Store(tid.x * 4, 42);
}

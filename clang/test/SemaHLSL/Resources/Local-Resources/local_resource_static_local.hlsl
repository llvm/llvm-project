// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// Test that a static local resource variable initialized from a global
// is accepted by both compilers.
//
// DXC: passes (both sema and codegen) with RWByteAddressBuffer.
//      Note: DXC asserts/ICEs when Texture2D is used instead.

RWByteAddressBuffer gBuf0 : register(u0);

uint Pass_StaticLocal(uint idx) {
    static RWByteAddressBuffer buf = gBuf0;
    buf.Store(idx * 4, 1);

    return 1;
}

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    Pass_StaticLocal(tid.x);
}

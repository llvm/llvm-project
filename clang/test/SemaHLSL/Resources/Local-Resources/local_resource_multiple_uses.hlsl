// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// Test that the same local resource can be passed to multiple helper
// functions within a single shader.
//
// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBuf0 : register(u0);

void HelperA(RWByteAddressBuffer buf, uint offset) {
    buf.Store(offset, 1);
}

void HelperB(RWByteAddressBuffer buf, uint offset) {
    buf.Store(offset, 2);
}

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    RWByteAddressBuffer local = gBuf0;
    HelperA(local, tid.x * 4);
    HelperB(local, tid.x * 4 + 4);
}

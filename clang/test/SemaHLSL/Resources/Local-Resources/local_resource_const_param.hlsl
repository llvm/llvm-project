// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// Test that a local resource can be passed as a const parameter.
//
// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBuf0 : register(u0);

void ReadOnly(const RWByteAddressBuffer buf, uint offset, out uint result) {
    result = 0;
}

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    RWByteAddressBuffer local = gBuf0;
    uint r;
    ReadOnly(local, tid.x * 4, r);
}

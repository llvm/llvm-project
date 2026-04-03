// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// Test that a local resource can be written through an out parameter.
//
// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBuf0 : register(u0);

void WriteThrough(out RWByteAddressBuffer buf) {
    buf = gBuf0;
}

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    RWByteAddressBuffer local;
    WriteThrough(local);
    local.Store(tid.x * 4, 42);
}

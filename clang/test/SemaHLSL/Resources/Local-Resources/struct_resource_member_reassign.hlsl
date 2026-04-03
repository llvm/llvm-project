// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// Test that reassigning the resource member of a local struct to a
// different global compiles without error.
//
// DXC: passes (both sema and codegen).
// Note: clang does not warn here because -Whlsl-explicit-binding
// tracks local resource variables, not struct member assignments.

RWByteAddressBuffer gBuf0 : register(u0);
RWByteAddressBuffer gBuf1 : register(u1);

struct ResHolder { RWByteAddressBuffer buf; };

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    ResHolder h;
    h.buf = gBuf0;
    h.buf = gBuf1;
    h.buf.Store(tid.x * 4, 42);
}

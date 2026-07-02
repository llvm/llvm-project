// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// Test that using a comma expression to initialize a local resource
// produces a warning about the unused left operand. The comma operator
// evaluates gBuf0, discards it, then uses gBuf1 as the initializer.
//
// DXC: passes (both sema and codegen), no diagnostic.

RWByteAddressBuffer gBuf0 : register(u0);
RWByteAddressBuffer gBuf1 : register(u1);

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    // expected-warning@+1 {{left operand of comma operator has no effect}}
    RWByteAddressBuffer buf = (gBuf0, gBuf1);
    buf.Store(tid.x * 4, 42);
}

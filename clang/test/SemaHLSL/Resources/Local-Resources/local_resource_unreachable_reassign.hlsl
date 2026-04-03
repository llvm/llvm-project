// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// Test that resource reassignment after an early return (in dead code
// from the perspective of the early-return path) still triggers the
// -Whlsl-explicit-binding warning.
//
// DXC: passes silently (no diagnostics even with -Wall).

RWByteAddressBuffer gBuf0 : register(u0);
RWByteAddressBuffer gBuf1 : register(u1);

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    RWByteAddressBuffer buf = gBuf0;
    // expected-note@-1 {{variable 'buf' is declared here}}
    if (tid.x > 0) {
        buf.Store(0, 1);
        return;
    }
    buf = gBuf1;
    // expected-warning@-1 {{assignment of 'gBuf1' to local resource 'buf' is not to the same unique global resource}}
    buf.Store(tid.x * 4, 42);
}

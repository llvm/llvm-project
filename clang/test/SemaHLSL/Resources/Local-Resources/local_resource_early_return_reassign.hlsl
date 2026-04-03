// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// DXC compiles this without any diagnostics (even with -Wall).
// Clang emits a -Whlsl-explicit-binding warning for the reassignment
// to a different global after the early return; DXC does not.

RWByteAddressBuffer gBuf0 : register(u0);
RWByteAddressBuffer gBuf1 : register(u1);

uint Pass_EarlyReturn(bool cond, uint idx) {
    // expected-note@+1{{variable 'buf' is declared here}}
    RWByteAddressBuffer buf = gBuf0;

    if (cond)
        buf.Store(idx * 4, 31);

        return 31;
    // DXC: no diagnostic. Clang: warning.
    // expected-warning@+1{{assignment of 'gBuf1' to local resource 'buf' is not to the same unique global resource}}
    buf = gBuf1;
    buf.Store(idx * 4, 31);

    return 31;
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x + tid.y * 8;
    Pass_EarlyReturn(true, idx);
}

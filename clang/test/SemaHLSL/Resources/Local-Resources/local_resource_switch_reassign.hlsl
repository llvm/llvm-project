// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// DXC compiles this without any diagnostics (even with -Wall).
// Clang emits -Whlsl-explicit-binding warnings for switch-case
// reassignments to different globals; DXC does not.

RWByteAddressBuffer gBuf0 : register(u0);
RWByteAddressBuffer gBuf1 : register(u1);
RWByteAddressBuffer gBuf2 : register(u2);

uint Pass_Switch(int v, uint idx) {
    // expected-note@+2{{variable 'buf' is declared here}}
    // expected-note@+1{{variable 'buf' is declared here}}
    RWByteAddressBuffer buf = gBuf0;

    switch (v) {
        // DXC: no diagnostic. Clang: warning.
        // expected-warning@+1{{assignment of 'gBuf1' to local resource 'buf' is not to the same unique global resource}}
        case 1: buf = gBuf1; break;
        // DXC: no diagnostic. Clang: warning.
        // expected-warning@+1{{assignment of 'gBuf2' to local resource 'buf' is not to the same unique global resource}}
        case 2: buf = gBuf2; break;
    }

    buf.Store(idx * 4, 20);


    return 20;
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x + tid.y * 8;
    Pass_Switch(1, idx);
}

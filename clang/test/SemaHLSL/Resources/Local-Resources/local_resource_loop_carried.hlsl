// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// DXC compiles this without any diagnostics (even with -Wall).
// Clang emits a -Whlsl-explicit-binding warning for the loop-carried
// reassignment from a different array element; DXC does not.

RWByteAddressBuffer gBuf0 : register(u0);
RWByteAddressBuffer gBufArray[4] : register(u10);

uint Pass_LoopCarried(int iterations, uint idx) {
    // expected-note@+1{{variable 'buf' is declared here}}
    RWByteAddressBuffer buf = gBuf0;

    for (int i=0;i<iterations;i++)
        // DXC: no diagnostic. Clang: warning.
        // expected-warning@+1{{assignment of 'gBufArray[i & 3]' to local resource 'buf' is not to the same unique global resource}}
        buf = gBufArray[i & 3];

    buf.Store(idx * 4, 26);


    return 26;
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x + tid.y * 8;
    Pass_LoopCarried(15, idx);
}

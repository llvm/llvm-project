// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// DXC compiles this without any diagnostics (even with -Wall).
// Clang emits a -Whlsl-explicit-binding warning for the ternary
// resource assignment inside the if-branch; DXC does not.

RWByteAddressBuffer gBuf0 : register(u0);
RWByteAddressBuffer gBuf1 : register(u1);
RWByteAddressBuffer gBuf2 : register(u2);

uint Pass_DeepPhi(bool a, bool b, uint idx) {
    RWByteAddressBuffer buf;

    if (a)
        // DXC: no diagnostic. Clang: warning.
        // expected-warning@+1{{assignment of 'b ? gBuf0 : gBuf1' to local resource 'buf' is not to the same unique global resource}}
        buf = b ? gBuf0 : gBuf1;
    else
        buf = gBuf2;

    buf.Store(idx * 4, 25);


    return 25;
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x + tid.y * 8;
    Pass_DeepPhi(true, false, idx);
}

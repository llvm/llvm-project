// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// DXC compiles this without any diagnostics (even with -Wall).
// Clang emits a -Whlsl-explicit-binding warning for the nested block
// reassignment to a different global; DXC does not.

RWByteAddressBuffer gBuf1 : register(u1);
RWByteAddressBuffer gBuf2 : register(u2);

uint Pass_NestedBlocks(uint idx) {
    // expected-note@+1{{variable 'buf' is declared here}}
    RWByteAddressBuffer buf;

    {
        buf = gBuf1;
        {
            // DXC: no diagnostic. Clang: warning.
            // expected-warning@+1{{assignment of 'gBuf2' to local resource 'buf' is not to the same unique global resource}}
            buf = gBuf2;
        }
    }

    buf.Store(idx * 4, 32);


    return 32;
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x + tid.y * 8;
    Pass_NestedBlocks(idx);
}

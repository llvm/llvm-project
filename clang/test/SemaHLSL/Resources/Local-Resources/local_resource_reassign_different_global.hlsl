// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// DXC compiles this without any diagnostics (even with -Wall).
// Clang emits -Whlsl-explicit-binding warnings for reassignment
// to a different global; DXC does not.
//
// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBuf0 : register(u0);
RWByteAddressBuffer gBuf1 : register(u1);

uint Pass_Reassign(uint idx) {
    // expected-note@+1{{variable 'buf' is declared here}}
    RWByteAddressBuffer buf = gBuf0;
    // expected-warning@+1{{assignment of 'gBuf1' to local resource 'buf' is not to the same unique global resource}}
    buf = gBuf1;
    buf.Store(idx * 4, 13);

    return 13;
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x + tid.y * 8;
    Pass_Reassign(idx);
}

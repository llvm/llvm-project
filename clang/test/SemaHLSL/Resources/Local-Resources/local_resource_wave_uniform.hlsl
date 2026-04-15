// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// Test that a wave-conditional resource reassignment produces a warning
// in clang but still generates valid IR.
//
// DXC: passes sema but fails codegen with:
//   "local resource not guaranteed to map to unique global resource."

RWByteAddressBuffer gBuf0 : register(u0);
RWByteAddressBuffer gBuf1 : register(u1);

uint Fail_WaveUniform(uint offset, uint value) {
    // expected-note@+1{{variable 'buf' is declared here}}
    RWByteAddressBuffer buf = gBuf0;
    if (WaveActiveAllTrue(true))
        // expected-warning@+1{{assignment of 'gBuf1' to local resource 'buf' is not to the same unique global resource}}
        buf = gBuf1;
    buf.Store(offset, value);

    return value;
}

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    Fail_WaveUniform(tid.x * 4, 10);
}

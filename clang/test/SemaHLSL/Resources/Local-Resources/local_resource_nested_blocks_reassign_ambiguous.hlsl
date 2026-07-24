// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

RWByteAddressBuffer GBuf1 : register(u1);
RWByteAddressBuffer GBuf2 : register(u2);

uint Pass_NestedBlocks(uint Cond, uint Idx) {
// expected-note@+1 {{variable 'Buf' is declared here}}
    RWByteAddressBuffer Buf;
    {
        Buf = GBuf1;
        if (Cond) {
// expected-warning@+1 {{assignment of 'GBuf2' to local resource 'Buf' is not to the same unique global resource}}
            Buf = GBuf2;
        }
    }
    Buf.Store(Idx * 4, 32);
    return 32;
}

[numthreads(8,8,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    uint Idx = Tid.x + Tid.y * 8;
    Pass_NestedBlocks(Tid.x & 1, Idx);
}

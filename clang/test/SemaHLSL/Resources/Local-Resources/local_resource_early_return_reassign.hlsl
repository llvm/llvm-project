// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

RWByteAddressBuffer GBuf0 : register(u0);
RWByteAddressBuffer GBuf1 : register(u1);

void Pass_EarlyReturn(bool Cond, uint Idx) {
// expected-note@+1 {{variable 'Buf' is declared here}}
    RWByteAddressBuffer Buf = GBuf0;

    if (Cond)
        Buf.Store(Idx * 4, 31);

        return;
// expected-warning@+1 {{assignment of 'GBuf1' to local resource 'Buf' is not to the same unique global resource}}
    Buf = GBuf1;
    Buf.Store(Idx * 4, 31);
}

[numthreads(8,8,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    uint Idx = Tid.x + Tid.y * 8;
    Pass_EarlyReturn(true, Idx);
}

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

RWByteAddressBuffer GBuf0 : register(u0);
RWByteAddressBuffer GBuf1 : register(u1);
RWByteAddressBuffer GBuf2 : register(u2);

void Pass_Switch(int V, uint Idx) {
// expected-note@+1 2 {{variable 'Buf' is declared here}}
    RWByteAddressBuffer Buf = GBuf0;
    switch (V) {
// expected-warning@+1 {{assignment of 'GBuf1' to local resource 'Buf' is not to the same unique global resource}}
        case 1: Buf = GBuf1; break;
// expected-warning@+1 {{assignment of 'GBuf2' to local resource 'Buf' is not to the same unique global resource}}
        case 2: Buf = GBuf2; break;
    }
    Buf.Store(Idx * 4, 20);
}

[numthreads(8,8,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    uint Idx = Tid.x + Tid.y * 8;
    Pass_Switch(int(Tid.x), Idx);
}

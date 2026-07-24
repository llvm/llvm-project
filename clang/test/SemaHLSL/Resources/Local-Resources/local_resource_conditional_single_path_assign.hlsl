// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics
RWByteAddressBuffer GBuf : register(u0);

[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    RWByteAddressBuffer Buf;
    if (Tid.x == 0)
        Buf = GBuf;
    Buf.Store(0, 42);
}

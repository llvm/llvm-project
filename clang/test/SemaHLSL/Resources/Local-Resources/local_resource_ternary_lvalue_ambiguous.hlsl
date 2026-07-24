// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics
RWByteAddressBuffer GBuf0 : register(u0);
RWByteAddressBuffer GBuf1 : register(u1);

[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    RWByteAddressBuffer A = GBuf0;
    RWByteAddressBuffer B = GBuf1;
    bool Cond = Tid.x > 0;
    (Cond ? A : B) = GBuf0;
    A.Store(Tid.x * 4, 1);
    B.Store(Tid.x * 4, 2);
}

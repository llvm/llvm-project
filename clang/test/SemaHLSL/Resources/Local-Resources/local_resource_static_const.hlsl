// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics
RWByteAddressBuffer GBuf0 : register(u0);

void Fail_StaticConst(uint Idx) {
    static const RWByteAddressBuffer Buf = GBuf0;
    Buf.Load(Idx * 4);
}

[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    Fail_StaticConst(Tid.x);
}

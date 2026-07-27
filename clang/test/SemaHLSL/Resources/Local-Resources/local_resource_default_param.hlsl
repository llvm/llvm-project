// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

RWByteAddressBuffer GBuf0 : register(u0);

// expected-error@+1 {{missing default argument on parameter 'Offset'}}
void Fail_DefaultParam(RWByteAddressBuffer Buf = GBuf0, uint Offset)
{
    Buf.Store(Offset, 42);
}

[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    Fail_DefaultParam(GBuf0, Tid.x * 4);
}

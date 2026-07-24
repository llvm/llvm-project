// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

RWByteAddressBuffer G0 : register(u0);

[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID)
{
// expected-error@+1 {{'register' attribute only applies to cbuffer/tbuffer and external global variables}}
    RWByteAddressBuffer Buf : register(u5) = G0;
    Buf.Store(Tid.x * 4, 42);
}

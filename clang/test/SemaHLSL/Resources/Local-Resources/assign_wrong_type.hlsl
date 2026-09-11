// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-note@*:* {{candidate function not viable: no known conversion from 'RWStructuredBuffer<uint>' (aka 'RWStructuredBuffer<unsigned int>') to 'const hlsl::RWByteAddressBuffer' for 1st argument}}
RWByteAddressBuffer GBuf0 : register(u0);
RWStructuredBuffer<uint> GSB : register(u1);

[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    RWByteAddressBuffer Buf = GBuf0;
    RWStructuredBuffer<uint> Sb = GSB;
// expected-error@+1 {{no viable overloaded '='}}
    Buf = Sb;
}

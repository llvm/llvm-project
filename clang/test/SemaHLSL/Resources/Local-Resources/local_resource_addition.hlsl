// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

RWByteAddressBuffer GBuf0 : register(u0);

float Fail_Add() {
    RWByteAddressBuffer Buf = GBuf0;
// expected-error@+1 {{invalid operands to binary expression ('RWByteAddressBuffer' and 'RWByteAddressBuffer')}}
    return Buf + Buf;
}

[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    Fail_Add();
}

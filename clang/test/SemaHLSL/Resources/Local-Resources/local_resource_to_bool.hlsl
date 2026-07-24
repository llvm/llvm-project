// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

RWByteAddressBuffer GBuf0 : register(u0);

bool Fail_Bool() {
    RWByteAddressBuffer Buf = GBuf0;
// expected-error@+1 {{no viable conversion from returned value of type 'RWByteAddressBuffer' to function return type 'bool'}}
    return Buf;
}

[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    Fail_Bool();
}

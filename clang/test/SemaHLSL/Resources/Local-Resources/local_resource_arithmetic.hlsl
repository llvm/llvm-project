// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

RWByteAddressBuffer GBuf0 : register(u0);

float Fail_Arithmetic() {
    RWByteAddressBuffer Buf = GBuf0;
// expected-error@+1 {{invalid operands to binary expression ('RWByteAddressBuffer' and 'int')}}
    Buf = Buf + 1;
// expected-error@+1 {{no viable conversion from returned value of type 'RWByteAddressBuffer' to function return type 'float'}}
    return Buf;
}

[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    Fail_Arithmetic();
}

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

RWByteAddressBuffer Out : register(u0);

[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
// expected-error@+1 {{too few initializers in list for type 'RWByteAddressBuffer[2]' (expected 2 but found 1)}}
    RWByteAddressBuffer Arr[2] = {Out};
    Arr[Tid.x & 1].Store(0, 42);
}

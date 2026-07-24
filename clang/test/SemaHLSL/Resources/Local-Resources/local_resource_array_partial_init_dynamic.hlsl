// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics
RWByteAddressBuffer Out : register(u0);

[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    RWByteAddressBuffer Arr[4];
    Arr[0] = Out;
    Arr[1] = Out;
    Arr[Tid.x & 3].Store(0, 42);
}

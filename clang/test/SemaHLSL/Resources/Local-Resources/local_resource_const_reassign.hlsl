// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-note@*:* {{candidate function not viable: 'this' argument has type 'const RWByteAddressBuffer', but method is not marked const}}
RWByteAddressBuffer GBuf0 : register(u0);
RWByteAddressBuffer GBuf1 : register(u1);

[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
// expected-note@+1 {{variable 'Buf' is declared here}}
    const RWByteAddressBuffer Buf = GBuf0;
// expected-warning@+2 {{assignment of 'GBuf1' to local resource 'Buf' is not to the same unique global resource}}
// expected-error@+1 {{no viable overloaded '='}}
    Buf = GBuf1;
}

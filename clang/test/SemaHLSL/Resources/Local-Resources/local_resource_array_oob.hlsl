// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-note@+1 {{array 'GBufArray' declared here}}
RWByteAddressBuffer GBufArray[4] : register(u0);

[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
// expected-warning@+1 {{array index 5 is past the end of the array (that has type 'RWByteAddressBuffer[4]')}}
    RWByteAddressBuffer Buf = GBufArray[5];
    
    Buf.Store(Tid.x * 4, 42);
}

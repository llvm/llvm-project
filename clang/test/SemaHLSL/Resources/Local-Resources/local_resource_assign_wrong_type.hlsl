// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// Test that assigning between incompatible resource types is rejected.
//
// DXC: also fails sema with a different message:
//   "cannot convert from 'RWStructuredBuffer<uint>' to 'RWByteAddressBuffer'"

RWByteAddressBuffer gBuf0 : register(u0);
RWStructuredBuffer<uint> gSB : register(u1);

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    RWByteAddressBuffer buf = gBuf0;
    RWStructuredBuffer<uint> sb = gSB;
    buf = sb;
    // expected-error@-1 {{no viable overloaded '='}}
    // expected-note@*:* {{candidate function not viable: no known conversion from 'RWStructuredBuffer<uint>' (aka 'RWStructuredBuffer<unsigned int>') to 'const hlsl::RWByteAddressBuffer' for 1st argument}}
}

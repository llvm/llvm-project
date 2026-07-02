// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// Test that C-style casting a SamplerState to RWByteAddressBuffer is rejected.
//
// DXC: also fails sema with a different message:
//   "cannot convert from 'SamplerState' to 'RWByteAddressBuffer'"

RWByteAddressBuffer gBuf0 : register(u0);
SamplerState gSampler : register(s0);

uint Fail_Reinterpret(uint offset, uint value) {
    RWByteAddressBuffer buf = gBuf0;
    ((RWByteAddressBuffer)gSampler).Store(offset, value);
    // expected-error@-1 {{no matching conversion for C-style cast from 'SamplerState' to 'RWByteAddressBuffer'}}
    // expected-note@*:* {{candidate constructor not viable: no known conversion from 'SamplerState' to 'const hlsl::RWByteAddressBuffer' for 1st argument}}
    // expected-note@*:* {{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
    return value;
}

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    Fail_Reinterpret(tid.x * 4, 8);
}

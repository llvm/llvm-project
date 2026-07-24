// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-note@*:* {{candidate constructor not viable: no known conversion from 'SamplerState' to 'const hlsl::RWByteAddressBuffer' for 1st argument}}
// expected-note@*:* {{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
RWByteAddressBuffer GBuf0 : register(u0);
SamplerState GSampler : register(s0);

uint Fail_Reinterpret(uint Offset, uint Value) {
    RWByteAddressBuffer Buf = GBuf0;
// expected-error@+1 {{no matching conversion for C-style cast from 'SamplerState' to 'RWByteAddressBuffer'}}
    ((RWByteAddressBuffer)GSampler).Store(Offset, Value);
    return Value;
}

[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    Fail_Reinterpret(Tid.x * 4, 8);
}

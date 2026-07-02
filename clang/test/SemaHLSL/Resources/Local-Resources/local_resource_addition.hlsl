// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// Test that adding two resource handles together is rejected.
//
// DXC: also fails sema with a different message:
//   "scalar, vector, or matrix expected"

RWByteAddressBuffer gBuf0 : register(u0);

float Fail_Add() {
    RWByteAddressBuffer buf = gBuf0;
    return buf + buf;
    // expected-error@-1 {{invalid operands to binary expression ('RWByteAddressBuffer' and 'RWByteAddressBuffer')}}
}

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    Fail_Add();
}

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// Test that comparing two resource handles with == is rejected.
//
// DXC: also fails sema with a different message:
//   "operator cannot be used with built-in type 'RWByteAddressBuffer'"

RWByteAddressBuffer gBuf0 : register(u0);
RWByteAddressBuffer gBuf1 : register(u1);

bool Fail_Compare() {
    RWByteAddressBuffer a = gBuf0;
    RWByteAddressBuffer b = gBuf1;
    return a == b;
    // expected-error@-1 {{invalid operands to binary expression ('RWByteAddressBuffer' and 'RWByteAddressBuffer')}}
}

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    Fail_Compare();
}

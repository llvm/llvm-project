// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// Test that C-style cast from a resource handle to uint is rejected.
//
// DXC: also fails sema with a different message:
//   "cannot convert from 'RWByteAddressBuffer' to 'uint'"

RWByteAddressBuffer gBuf0 : register(u0);

uint Fail_Cast() {
    RWByteAddressBuffer buf = gBuf0;
    return (uint)buf;
    // expected-error@-1 {{cannot convert 'RWByteAddressBuffer' to 'uint' (aka 'unsigned int') without a conversion operator}}
}

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    Fail_Cast();
}

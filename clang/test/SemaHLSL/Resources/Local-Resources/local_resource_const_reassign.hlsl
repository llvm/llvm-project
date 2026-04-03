// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// Test that a const-qualified local resource cannot be reassigned.
//
// DXC: also fails sema with:
//   "cannot assign to variable 'buf' with const-qualified type
//    'const RWByteAddressBuffer'"

RWByteAddressBuffer gBuf0 : register(u0);
RWByteAddressBuffer gBuf1 : register(u1);

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    const RWByteAddressBuffer buf = gBuf0;
    buf = gBuf1;
    // expected-warning@-1 {{assignment of 'gBuf1' to local resource 'buf' is not to the same unique global resource}}
    // expected-note@-3 {{variable 'buf' is declared here}}
    // expected-error@-3 {{no viable overloaded '='}}
    // expected-note@*:* {{candidate function not viable: 'this' argument has type 'const RWByteAddressBuffer', but method is not marked const}}
}

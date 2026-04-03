// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// Test that the volatile qualifier on a local resource prevents calling
// methods on it, because the methods are not marked volatile.
//
// DXC: accepts volatile on resources and compiles successfully.

RWByteAddressBuffer gBuf0 : register(u0);

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    volatile RWByteAddressBuffer buf = gBuf0;
    // expected-note@*:* {{candidate function template not viable: 'this' argument has type 'volatile RWByteAddressBuffer', but method is not marked volatile}}
    // expected-note@*:* {{candidate function not viable: 'this' argument has type 'volatile RWByteAddressBuffer', but method is not marked volatile}}
    // expected-error@+1 {{no matching member function for call to 'Store'}}
    buf.Store(tid.x * 4, 42);
}

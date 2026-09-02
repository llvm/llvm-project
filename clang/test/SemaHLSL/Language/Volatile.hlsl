// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -verify %s

RWByteAddressBuffer gBuf : register(u0);

[numthreads(1,1,1)]
void main() {
    // expected-error@+1 {{unknown type name 'volatile'}}
    volatile RWByteAddressBuffer buf = gBuf; // expected-error {{expected ';' at end of declaration}}
    buf.Store(0, 42); // expected-error {{use of undeclared identifier 'buf'}}
}

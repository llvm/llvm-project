// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-compute -x hlsl -fsyntax-only %s -verify
// This test validates that initializing a static local variable with a resource
// type produces a diagnostic error instead of crashing during optimization.

RWByteAddressBuffer gBuf0 : register(u0);

void fn() {
    // expected-error@+1 {{static local resource variable is not allowed}}
    static RWByteAddressBuffer buf = gBuf0; 
}

[numthreads(1,1,1)]
void main() {
}

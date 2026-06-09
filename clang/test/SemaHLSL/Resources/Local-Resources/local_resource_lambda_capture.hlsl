// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// Test that a lambda capturing a local resource by value cannot call
// non-const methods on the captured resource. The lambda's call operator
// is implicitly const, so the captured `buf` is treated as const, and
// `Store` is not marked const.
//
// DXC: does not support lambdas at all — rejects with a parse error
// "expected expression" at the '[' of the capture list.

RWByteAddressBuffer g0 : register(u0);

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    RWByteAddressBuffer buf = g0;
    // expected-warning@+2 {{'auto' type specifier is a HLSL 202y extension}}
    // expected-warning@+1 {{lambdas are a clang HLSL extension}}
    auto fn = [=]() { buf.Store(0, 42); };
    // expected-error@-1 {{no matching member function for call to 'Store'}}
    // expected-note@*:* {{candidate function template not viable: 'this' argument has type 'const RWByteAddressBuffer', but method is not marked const}}
    // expected-note@*:* {{candidate function not viable: 'this' argument has type 'const RWByteAddressBuffer', but method is not marked const}}
    fn();
}

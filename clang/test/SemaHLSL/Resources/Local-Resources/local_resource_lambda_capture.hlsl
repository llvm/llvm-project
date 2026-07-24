// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-note@*:* {{candidate function template not viable: 'this' argument has type 'const RWByteAddressBuffer', but method is not marked const}}
// expected-note@*:* {{candidate function not viable: 'this' argument has type 'const RWByteAddressBuffer', but method is not marked const}}
RWByteAddressBuffer G0 : register(u0);

[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID)
{
    RWByteAddressBuffer Buf = G0;
// expected-warning@+3 {{'auto' type specifier is a HLSL 202y extension}}
// expected-warning@+2 {{lambdas are a clang HLSL extension}}
// expected-error@+1 {{no matching member function for call to 'Store'}}
    auto Fn = [=]() { Buf.Store(0, 42); };
    Fn();
}

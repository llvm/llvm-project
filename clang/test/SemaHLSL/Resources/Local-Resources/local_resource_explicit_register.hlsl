// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// Test that an explicit `register()` attribute on a local resource variable
// is rejected. The register attribute only applies to global cbuffer/tbuffer
// and external global variables.
//
// DXC: silently *ignores* the register attribute on a local resource and
// compiles successfully (using the implicit binding from the global). This
// is a DXC quirk; clang correctly rejects.

RWByteAddressBuffer g0 : register(u0);

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    RWByteAddressBuffer buf : register(u5) = g0; // expected-error {{'register' attribute only applies to cbuffer/tbuffer and external global variables}}
    buf.Store(tid.x * 4, 42);
}

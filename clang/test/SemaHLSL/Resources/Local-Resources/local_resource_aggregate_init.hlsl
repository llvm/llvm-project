// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// Test that aggregate (brace) initialization of a struct with a resource
// member works correctly.
//
// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBuf0 : register(u0);

struct ResHolder { RWByteAddressBuffer buf; };

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    ResHolder h = {gBuf0};
    h.buf.Store(tid.x * 4, 42);
}

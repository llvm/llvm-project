// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// Test that a function can return a struct containing a resource member.
//
// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBuf0 : register(u0);

struct ResHolder { RWByteAddressBuffer buf; };

ResHolder MakeHolder() {
    ResHolder h;
    h.buf = gBuf0;
    return h;
}

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    ResHolder h = MakeHolder();
    h.buf.Store(tid.x * 4, 42);
}

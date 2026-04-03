// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// Test that a struct with both a resource member and a scalar member
// can be used as a local variable.
//
// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBuf0 : register(u0);

struct Node {
    RWByteAddressBuffer buf;
    uint next;
};

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    Node n;
    n.buf = gBuf0;
    n.next = 0;
    n.buf.Store(tid.x * 4, n.next);
}

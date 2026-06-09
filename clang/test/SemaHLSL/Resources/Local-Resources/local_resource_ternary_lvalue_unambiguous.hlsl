// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// Test that a ternary expression used as an lvalue with a statically
// constant condition is, in principle, statically resolvable to a
// single side (here `a`), but the front end still warns because its
// binding-ambiguity check is syntactic. The backend's optimizer can
// fold the constant condition away, so codegen succeeds cleanly.

RWByteAddressBuffer gBuf0 : register(u0);
RWByteAddressBuffer gBuf1 : register(u1);

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    RWByteAddressBuffer a = gBuf0;
    RWByteAddressBuffer b = gBuf1;
    // expected-warning@+1 {{lvalue ternary on local resources reassigns a binding-ambiguous handle}}
    (true ? a : b) = gBuf0;
    a.Store(tid.x * 4, 1);
    b.Store(tid.x * 4, 2);
}

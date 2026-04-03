// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// Test that using a default-initialized (unbound) local resource produces
// valid IR in clang. No warnings or errors from clang.
//
// DXC: passes (both sema and codegen).

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    RWByteAddressBuffer buf;
    buf.Store(tid.x * 4, 42);
}

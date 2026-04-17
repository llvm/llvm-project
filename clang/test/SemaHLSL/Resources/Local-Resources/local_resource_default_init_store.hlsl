// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// Test that using a default-initialized (unbound) local resource produces
// valid IR in clang. No warnings or errors from clang.
//
// DXC: error (codegen) — "local resource not guaranteed to map to unique
// global resource".
//
// Note: full DXIL codegen (not tested here) crashes in DXILOpLowering.
// Bug: https://github.com/llvm/llvm-project/issues/192551

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    RWByteAddressBuffer buf;
    buf.Store(tid.x * 4, 42);
}

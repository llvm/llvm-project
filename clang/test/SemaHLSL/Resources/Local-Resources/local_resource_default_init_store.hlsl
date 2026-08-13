// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify
// XFAIL: *

/// Storing through a local resource that was never assigned a global cannot
/// work — there is no binding for the access to lower to.
///
/// Clang's CFG-based uninitialized analysis already diagnoses this shape for
/// scalars, but resource types do not participate in it, so this compiles
/// silently. The expectations below use the wording that analysis already
/// emits, so this test flips to XPASS once resources are covered.
///
/// Tracking issue: https://github.com/llvm/llvm-project/issues/216193

[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
// expected-note@+1 {{initialize the variable 'Buf' to silence this warning}}
    RWByteAddressBuffer Buf;
// expected-warning@+1 {{variable 'Buf' is uninitialized when used here}}
    Buf.Store(Tid.x * 4, 42);
}
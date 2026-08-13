// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify
// XFAIL: *

/// The local is assigned on only one path, so the access is uninitialized
/// whenever the condition is false. The optimizer folds the undefined path
/// away, which makes the shader appear to work.
///
/// Clang's CFG-based uninitialized analysis already diagnoses this shape for
/// scalars, but resource types do not participate in it, so this compiles
/// silently. The expectations below use the wording that analysis already
/// emits, so this test flips to XPASS once resources are covered.
///
/// Tracking issue: https://github.com/llvm/llvm-project/issues/216193

RWByteAddressBuffer GBuf : register(u0);

[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
// expected-note@+1 {{initialize the variable 'Buf' to silence this warning}}
    RWByteAddressBuffer Buf;
// expected-warning@+1 {{variable 'Buf' is used uninitialized whenever 'if' condition is false}}
    if (Tid.x == 0)
        Buf = GBuf;
    Buf.Store(0, 42);
}
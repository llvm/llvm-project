// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics
[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    RWByteAddressBuffer Buf;
    Buf.Store(Tid.x * 4, 42);
}

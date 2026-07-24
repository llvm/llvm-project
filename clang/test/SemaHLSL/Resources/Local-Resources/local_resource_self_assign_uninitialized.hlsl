// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics
RWByteAddressBuffer GBuf : register(u0);

[numthreads(1,1,1)]
void main() {
    RWByteAddressBuffer Out;
    Out = Out;
    Out.Store(0, 42);
}

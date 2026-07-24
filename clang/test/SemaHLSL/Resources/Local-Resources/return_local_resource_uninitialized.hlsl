// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics
RWByteAddressBuffer GBuf : register(u0);

RWByteAddressBuffer GetUninitialized() {
    RWByteAddressBuffer Buf;
    return Buf;
}

[numthreads(1,1,1)]
void main() {
    RWByteAddressBuffer Tmp = GetUninitialized();
    Tmp.Store(0, 42);
}

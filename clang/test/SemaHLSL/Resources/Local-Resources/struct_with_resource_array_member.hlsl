// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

RWByteAddressBuffer GBuf : register(u0);

struct S { RWByteAddressBuffer Arr[2]; };

[numthreads(1,1,1)]
void main() {
    S S;
// expected-error@+1 {{no member named 'arr' in 'S'}}
    S.arr[0] = GBuf;
// expected-error@+1 {{no member named 'arr' in 'S'}}
    S.arr[0].Store(0, 42);
}

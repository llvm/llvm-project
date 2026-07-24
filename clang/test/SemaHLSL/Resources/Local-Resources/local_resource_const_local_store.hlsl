// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-note@*:* 2 {{candidate function template not viable: 'this' argument has type 'const RWByteAddressBuffer', but method is not marked const}}
// expected-note@*:* 2 {{candidate function not viable: 'this' argument has type 'const RWByteAddressBuffer', but method is not marked const}}
RWByteAddressBuffer GBuf : register(u0);

uint LoadConst(const RWByteAddressBuffer Buf) {
    return Buf.Load(0);
}

[numthreads(1,1,1)]
void main() {
    const RWByteAddressBuffer Local = GBuf;
// expected-error@+1 {{no matching member function for call to 'Store'}}
    Local.Store(0, 99);
    uint Val = LoadConst(Local);
// expected-error@+1 {{no matching member function for call to 'Store'}}
    Local.Store(0, Val == 99 ? 42 : 0);
}

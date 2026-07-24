// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-note@*:* {{candidate function not viable: 'this' argument has type 'const RWByteAddressBuffer', but method is not marked const}}
// expected-note@*:* {{candidate function template not viable: 'this' argument has type 'const RWByteAddressBuffer', but method is not marked const}}
RWByteAddressBuffer GBuf : register(u0);

void StoreInCallee(const RWByteAddressBuffer Buf, uint V) {
// expected-error@+1 {{no matching member function for call to 'Store'}}
    Buf.Store(0, V);
}

[numthreads(1,1,1)]
void main() {
    StoreInCallee(GBuf, 42);
}

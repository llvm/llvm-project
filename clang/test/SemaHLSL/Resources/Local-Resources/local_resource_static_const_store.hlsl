// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-note@*:* {{candidate function not viable: 'this' argument has type 'const RWByteAddressBuffer', but method is not marked const}}
// expected-note@*:* {{candidate function template not viable: 'this' argument has type 'const RWByteAddressBuffer', but method is not marked const}}
RWByteAddressBuffer GBuf0 : register(u0);

void Fail_StaticConstStore(uint Idx, uint Value) {
    static const RWByteAddressBuffer Buf = GBuf0;
// expected-error@+1 {{no matching member function for call to 'Store'}}
    Buf.Store(Idx * 4, Value);
}

[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    Fail_StaticConstStore(Tid.x, 42);
}

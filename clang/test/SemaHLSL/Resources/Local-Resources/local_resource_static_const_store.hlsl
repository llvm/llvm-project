// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// Test that a `static const` local resource cannot have Store called on it,
// because Store is not marked const. This complements
// local_resource_static_const.hlsl which tests the Load side of the same
// const-method-mismatch check.
//
// DXC: ICEs with "llvm::cast<X>() argument of incompatible type!" on this
// pattern, just like the Load variant.

RWByteAddressBuffer gBuf0 : register(u0);

void Fail_StaticConstStore(uint idx, uint value) {
    static const RWByteAddressBuffer buf = gBuf0;
    // expected-note@*:* {{candidate function not viable: 'this' argument has type 'const RWByteAddressBuffer', but method is not marked const}}
    // expected-note@*:* {{candidate function template not viable: 'this' argument has type 'const RWByteAddressBuffer', but method is not marked const}}
    // expected-error@+1 {{no matching member function for call to 'Store'}}
    buf.Store(idx * 4, value);
}

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    Fail_StaticConstStore(tid.x, 42);
}

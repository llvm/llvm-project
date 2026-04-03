// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// Test that a static const local resource cannot call any methods,
// because neither Load nor Store is marked const. This combines the
// static and const qualifiers, which are tested separately in
// local_resource_static_local.hlsl and local_resource_const_param.hlsl.
//
// DXC: ICEs with "llvm::cast<X>() argument of incompatible type!"

RWByteAddressBuffer gBuf0 : register(u0);

uint Fail_StaticConst(uint idx) {
    static const RWByteAddressBuffer buf = gBuf0;
    // expected-note@*:* {{candidate function not viable: 'this' argument has type 'const RWByteAddressBuffer', but method is not marked const}}
    // expected-note@*:* {{candidate template ignored: couldn't infer template argument 'element_type'}}
    // expected-note@*:* {{candidate function not viable: requires 2 arguments, but 1 was provided}}
    // expected-note@*:* {{candidate function template not viable: requires 2 arguments, but 1 was provided}}
    // expected-error@+1 {{no matching member function for call to 'Load'}}
    return buf.Load(idx * 4);
}

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    Fail_StaticConst(tid.x);
}

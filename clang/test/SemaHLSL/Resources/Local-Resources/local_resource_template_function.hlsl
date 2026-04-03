// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// Test that a template function can accept and use a resource parameter.
// This exercises template instantiation with resource types.
//
// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBuf0 : register(u0);

template<typename T>
void UseResource(T buf, uint idx, uint val) {
    buf.Store(idx * 4, val);
}

uint Pass_TemplateFunction(uint idx) {
    RWByteAddressBuffer buf = gBuf0;
    UseResource(buf, idx, 42);
    return 42;
}

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    Pass_TemplateFunction(tid.x);
}

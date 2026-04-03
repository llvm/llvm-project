// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// Test that a method can be called directly on the return value of a
// function that returns a resource. This exercises temporary resource
// lifetime — the method is invoked on an rvalue.
//
// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBuf0 : register(u0);

RWByteAddressBuffer GetBuf() { return gBuf0; }

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    GetBuf().Store(tid.x * 4, 42);
}

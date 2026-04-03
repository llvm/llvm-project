// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// Test that RWStructuredBuffer<uint> works as a local variable with
// subscript operator access. This is distinct from RWByteAddressBuffer
// because it uses typed element access via operator[].
//
// DXC: passes (both sema and codegen).

RWStructuredBuffer<uint> gSB : register(u0);

uint Pass_StructuredBufferLocal(uint idx) {
    RWStructuredBuffer<uint> sb = gSB;
    sb[idx] = 42;
    return 42;
}

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    Pass_StructuredBufferLocal(tid.x);
}

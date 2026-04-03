// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// Test that function overloading works when overloads differ by
// resource type parameter. This exercises overload resolution
// with resource types.
//
// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBuf0 : register(u0);
RWStructuredBuffer<uint> gSB : register(u1);

void DoStore(RWByteAddressBuffer buf, uint idx, uint val) {
    buf.Store(idx * 4, val);
}

void DoStore(RWStructuredBuffer<uint> buf, uint idx, uint val) {
    buf[idx] = val;
}

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    RWByteAddressBuffer localBuf = gBuf0;
    RWStructuredBuffer<uint> localSB = gSB;
    DoStore(localBuf, tid.x, 1);
    DoStore(localSB, tid.x, 2);
}

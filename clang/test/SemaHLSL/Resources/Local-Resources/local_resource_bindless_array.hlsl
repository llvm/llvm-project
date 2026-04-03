// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBufArray[4] : register(u10);

uint Pass_Bindless(uint idx) {
    RWByteAddressBuffer buf = gBufArray[idx & 3];
    buf.Store(idx * 4, 21);

    return 21;
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x + tid.y * 8;
    Pass_Bindless(idx);
}

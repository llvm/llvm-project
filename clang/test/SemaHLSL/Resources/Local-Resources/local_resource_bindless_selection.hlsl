// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBufArray[4] : register(u10);

uint Pass_BindlessSelection(uint a, uint b, uint idx) {
    RWByteAddressBuffer buf;

    buf = gBufArray[a & 3];
    buf = gBufArray[b & 3];

    buf.Store(idx * 4, 33);


    return 33;
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x + tid.y * 8;
    Pass_BindlessSelection(2, 3, idx);
}

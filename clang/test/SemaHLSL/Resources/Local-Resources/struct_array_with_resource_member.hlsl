// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBuf0 : register(u0);

struct BufStruct { RWByteAddressBuffer buf; };

uint Pass_StructArray(uint idx) {
    BufStruct s[2];
    s[0].buf = gBuf0;
    s[0].buf.Store(idx * 4, 5);

    return 5;
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x + tid.y * 8;
    Pass_StructArray(idx);
}

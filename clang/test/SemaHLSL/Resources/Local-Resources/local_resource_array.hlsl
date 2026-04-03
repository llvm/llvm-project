// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBuf0 : register(u0);

uint Pass_LocalArray(uint idx) {
    RWByteAddressBuffer arr[2];
    arr[0] = gBuf0;
    arr[0].Store(idx * 4, 10);

    return 10;
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x + tid.y * 8;
    Pass_LocalArray(idx);
}

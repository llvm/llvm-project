// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// DXC: passes (both sema and codegen).

uint Pass_Uninitialized(uint idx) {
    RWByteAddressBuffer buf;
    buf.Store(idx * 4, 11);

    return 11;
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x + tid.y * 8;
    Pass_Uninitialized(idx);
}

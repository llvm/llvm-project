// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBuf2 : register(u2);

struct ForwardA { RWByteAddressBuffer buf; };
struct ForwardB { ForwardA a; };

uint Pass_ForwardStructLayers(uint idx) {
    ForwardB b;
    b.a.buf = gBuf2;
    b.a.buf.Store(idx * 4, 29);

    return 29;
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x + tid.y * 8;
    Pass_ForwardStructLayers(idx);
}

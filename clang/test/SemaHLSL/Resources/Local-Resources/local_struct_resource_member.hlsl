// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBuf0 : register(u0);

struct PassStruct {
    RWByteAddressBuffer buf;
};

uint Pass_Struct(uint idx) {
    PassStruct s;
    s.buf = gBuf0;
    s.buf.Store(idx * 4, 16);

    return 16;
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x + tid.y * 8;
    Pass_Struct(idx);
}

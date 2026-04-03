// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBuf1 : register(u1);

uint Pass_Level2(RWByteAddressBuffer buf, uint idx) {
    buf.Store(idx*4, 17);
    return 17;
}

uint Pass_Level1(RWByteAddressBuffer buf, uint idx) {
    return Pass_Level2(buf, idx);
}

uint Pass_FunctionForward(uint idx) {
    RWByteAddressBuffer buf = gBuf1;
    return Pass_Level1(buf, idx);
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x + tid.y * 8;
    Pass_FunctionForward(idx);
}

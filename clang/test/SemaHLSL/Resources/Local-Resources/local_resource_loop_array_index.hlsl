// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBufArray[4] : register(u10);

uint Pass_Loop(uint idx) {
    uint sum = 0;
    for (unsigned int i=0;i<4;i++) {    
        RWByteAddressBuffer buf = gBufArray[i];
        buf.Store(idx * 4 + i * 4, 15);

        sum += 15;
    }
    return sum;
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x + tid.y * 8;
    Pass_Loop(idx);
}

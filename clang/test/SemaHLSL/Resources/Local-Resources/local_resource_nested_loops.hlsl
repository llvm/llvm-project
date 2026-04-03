// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBufArray[4] : register(u10);

uint Pass_NestedLoops(uint idx) {
    uint sum = 0;
    for (unsigned int i=0;i<2;i++)
    for (unsigned int j=0;j<2;j++) {        
        RWByteAddressBuffer buf = gBufArray[i+j];
        buf.Store(idx * 4 + (i+j)*4, 23);

        sum += 23;
    }
    return sum;
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x + tid.y * 8;
    Pass_NestedLoops(idx);
}

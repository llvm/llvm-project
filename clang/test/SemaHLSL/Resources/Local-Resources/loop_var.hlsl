// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// DXC: passes (both sema and codegen).

RWByteAddressBuffer gBuf0 : register(u0);

RWByteAddressBuffer gOut  : register(u3);

void Pass_LoopVar() {
    for (RWByteAddressBuffer buf = gBuf0; false == false; ) {
        buf.Store(0, 0);
        break; 
    }
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID) {    
    Pass_LoopVar();    
}

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics
RWByteAddressBuffer GOut  : register(u3);

uint DoStore(RWByteAddressBuffer Buf, uint Offset, uint Value) {
    Buf.Store(Offset, Value);
    return Value;
}

struct PassBufStruct { RWByteAddressBuffer Buf; };

groupshared PassBufStruct SharedStruct;

uint Use_PassSharedStruct(uint Idx) {
    return DoStore(SharedStruct.Buf, Idx * 4, 1);
}

[numthreads(8,8,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    uint Idx = Tid.x + Tid.y * 8;
    Use_PassSharedStruct(Idx);    
}

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -verify

RWByteAddressBuffer gOut  : register(u3);

uint DoStore(RWByteAddressBuffer buf, uint offset, uint value)
{
    buf.Store(offset, value);
    return value;
}

groupshared RWByteAddressBuffer sharedBuf;
uint Use_Shared(uint idx)
{
    return DoStore(sharedBuf, idx * 4, 1);
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x + tid.y * 8;
    Use_Shared(idx);    
}
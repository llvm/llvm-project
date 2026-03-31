// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -verify

// expected-no-diagnostics

RWByteAddressBuffer gBuf0 : register(u0);
RWByteAddressBuffer gBuf1 : register(u1);

RWByteAddressBuffer gOut  : register(u3);

uint DoStore(RWByteAddressBuffer buf, uint offset, uint value)
{
    buf.Store(offset, value);
    return value;
}

uint Pass_ExpressionInit(uint idx)
{
    RWByteAddressBuffer buf = (true ? gBuf0 : gBuf1);
    return DoStore(buf, idx * 4, 3);
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID)
{    
    uint idx = tid.x + tid.y * 8;
    Pass_ExpressionInit(idx);    
}
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// This test fails validation in DXC, but DXC's sema allows it.
// Meanwhile, it is for some reason accepted by clang's sema.
// TODO: Why does this pass clang's sema, but use_groupshared.hlsl fails clang's sema?
// Run a git bisect on https://github.com/llvm/llvm-project/issues/158107, and figure 
// out which commit seemed to resolve this issue.

RWByteAddressBuffer gOut  : register(u3);

// expected-note@+1{{passing argument to parameter 'buf' here}}
uint DoStore(RWByteAddressBuffer buf, uint offset, uint value)
{
    buf.Store(offset, value);
    return value;
}

// expected-note@*:*{{candidate constructor not viable: cannot bind reference in address space 'groupshared' to object in generic address space in 1st argument}}
// expected-note@*:*{{candidate constructor not viable: requires 0 arguments, but 1 was provided}}

struct PassBufStruct { RWByteAddressBuffer buf; };

groupshared PassBufStruct sharedStruct;

uint Use_PassSharedStruct(uint idx)
{
    // expected-error@+1{{no matching constructor for initialization of 'RWByteAddressBuffer'}}
    return DoStore(sharedStruct.buf, idx * 4, 1);
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x + tid.y * 8;
    Use_PassSharedStruct(idx);    
}

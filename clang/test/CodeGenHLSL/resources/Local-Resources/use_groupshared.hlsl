// RUN: not %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute -emit-llvm -disable-llvm-passes 2>&1 -o - %s | llvm-cxxfilt | FileCheck %s

// This test fails validation in DXC, but DXC's sema allows it.
// Meanwhile, groupshared is immediately detected and rejected in clang's sema.
//
// DXC: passes sema but fails validation with:
//   "Explicit load/store type does not match pointee type of pointer operand"

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

// CHECK: error: no matching constructor for initialization of 'RWByteAddressBuffer'
// CHECK: note: candidate constructor not viable: cannot bind reference in address space 'groupshared' to object in generic address space in 1st argument
// CHECK: note: candidate constructor not viable: requires 0 arguments, but 1 was provided
// CHECK: note: passing argument to parameter 'buf' here

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x + tid.y * 8;
    Use_Shared(idx);
}

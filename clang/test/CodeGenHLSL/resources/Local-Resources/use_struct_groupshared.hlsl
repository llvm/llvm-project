// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute -emit-llvm -disable-llvm-passes 2>&1 -o - %s | llvm-cxxfilt | FileCheck %s
//
// XFAIL: *
// https://github.com/llvm/llvm-project/issues/158107

// This test fails validation in DXC, but DXC's sema allows it.
// Meanwhile, it is for some reason accepted by clang's sema.
// TODO: Why does this pass clang's sema, but use_groupshared.hlsl fails clang's sema?
// Run a git bisect on https://github.com/llvm/llvm-project/issues/158107, and figure 
// out which commit seemed to resolve this issue.
//
// DXC: passes sema but fails validation with:
//   "Explicit load/store type does not match pointee type of pointer operand"

RWByteAddressBuffer gOut  : register(u3);

// CHECK: note: passing argument to parameter 'buf' here
uint DoStore(RWByteAddressBuffer buf, uint offset, uint value) {
    buf.Store(offset, value);
    return value;
}

// CHECK: note: candidate constructor not viable: cannot bind reference in address space 'groupshared' to object in generic address space in 1st argument
// CHECK: note: candidate constructor not viable: requires 0 arguments, but 1 was provided

struct PassBufStruct { RWByteAddressBuffer buf; };

groupshared PassBufStruct sharedStruct;

uint Use_PassSharedStruct(uint idx) {
    // CHECK: error: no matching constructor for initialization of 'RWByteAddressBuffer'
    return DoStore(sharedStruct.buf, idx * 4, 1);
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x + tid.y * 8;
    Use_PassSharedStruct(idx);    
}

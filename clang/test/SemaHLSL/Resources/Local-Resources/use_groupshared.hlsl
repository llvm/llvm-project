// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// This test fails validation in DXC, but DXC's sema allows it.
// Meanwhile, groupshared is immediately detected and rejected in clang's sema.
//
// DXC: passes sema but fails validation with:
//   "Explicit load/store type does not match pointee type of pointer operand"

RWByteAddressBuffer gOut  : register(u3);

// expected-note@+1{{passing argument to parameter 'buf' here}}
uint DoStore(RWByteAddressBuffer buf, uint offset, uint value) {
    buf.Store(offset, value);
    return value;
}

groupshared RWByteAddressBuffer sharedBuf;
uint Use_Shared(uint idx) {
    // expected-note@*:*{{candidate constructor not viable: cannot bind reference in address space 'groupshared'}}
    // expected-note@*:*{{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
    // expected-error@+1{{no matching constructor for initialization of 'RWByteAddressBuffer'}}
    return DoStore(sharedBuf, idx * 4, 1);
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x + tid.y * 8;
    Use_Shared(idx);
}

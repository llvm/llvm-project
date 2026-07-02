// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// Test that calling Store directly on a groupshared resource is rejected
// because the method expects a generic address space object.
//
// This is distinct from use_groupshared.hlsl, which tests passing a
// groupshared resource as a function argument (constructor mismatch).
//
// DXC: allows this at sema but rejects during validation.

groupshared RWByteAddressBuffer sharedBuf;

uint Use_SharedDirect(uint idx) {
    // expected-note@*:*{{candidate function template not viable: 'this' object is in address space 'groupshared', but method expects object in generic address space}}
    // expected-note@*:*{{candidate function not viable: 'this' object is in address space 'groupshared', but method expects object in generic address space}}
    // expected-error@+1{{no matching member function for call to 'Store'}}
    sharedBuf.Store(idx * 4, 1);
    return 1;
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x + tid.y * 8;
    Use_SharedDirect(idx);
}

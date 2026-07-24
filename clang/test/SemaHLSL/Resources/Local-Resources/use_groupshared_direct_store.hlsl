// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-note@*:* {{candidate function template not viable: 'this' object is in address space 'groupshared', but method expects object in generic address space}}
// expected-note@*:* {{candidate function not viable: 'this' object is in address space 'groupshared', but method expects object in generic address space}}
groupshared RWByteAddressBuffer SharedBuf;

uint Use_SharedDirect(uint Idx) {
// expected-error@+1 {{no matching member function for call to 'Store'}}
    SharedBuf.Store(Idx * 4, 1);
    return 1;
}

[numthreads(8,8,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    uint Idx = Tid.x + Tid.y * 8;
    Use_SharedDirect(Idx);
}

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-note@*:* {{candidate constructor not viable: cannot bind reference in address space 'groupshared' to object in generic address space in 1st argument}}
// expected-note@*:* {{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
RWByteAddressBuffer GOut  : register(u3);

// expected-note@+1 {{passing argument to parameter 'Buf' here}}
uint DoStore(RWByteAddressBuffer Buf, uint Offset, uint Value) {
    Buf.Store(Offset, Value);
    return Value;
}

groupshared RWByteAddressBuffer SharedBuf;
uint Use_Shared(uint Idx) {
// expected-error@+1 {{no matching constructor for initialization of 'RWByteAddressBuffer'}}
    return DoStore(SharedBuf, Idx * 4, 1);
}

[numthreads(8,8,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    uint Idx = Tid.x + Tid.y * 8;
    Use_Shared(Idx);
}

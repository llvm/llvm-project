// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// expected-no-diagnostics

// DXC: passes (both sema and codegen).

RWByteAddressBuffer Pass_ReturnLocal_Uninitialized() {
    RWByteAddressBuffer buf;
    return buf;
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    RWByteAddressBuffer tmp = Pass_ReturnLocal_Uninitialized();
}

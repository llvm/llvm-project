// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// Test that arithmetic operations on resource types are rejected.
//
// DXC: also fails sema, but with different messages:
//   "scalar, vector, or matrix expected" (for buf + 1)
//   "cannot initialize return object of type 'float' with an lvalue
//    of type 'RWByteAddressBuffer'" (for return buf)

RWByteAddressBuffer gBuf0 : register(u0);

float Fail_Arithmetic() {
    RWByteAddressBuffer buf = gBuf0;
    buf = buf + 1;
    // expected-error@-1 {{invalid operands to binary expression ('RWByteAddressBuffer' and 'int')}}
    return buf;
    // expected-error@-1 {{no viable conversion from returned value of type 'RWByteAddressBuffer' to function return type 'float'}}
}

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    Fail_Arithmetic();
}

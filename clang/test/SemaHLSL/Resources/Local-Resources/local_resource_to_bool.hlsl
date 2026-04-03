// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// Test that implicit conversion of a resource handle to bool is rejected.
//
// DXC: also fails sema with a different message:
//   "cannot initialize return object of type 'bool' with an lvalue
//    of type 'RWByteAddressBuffer'"

RWByteAddressBuffer gBuf0 : register(u0);

bool Fail_Bool() {
    RWByteAddressBuffer buf = gBuf0;
    return buf;
    // expected-error@-1 {{no viable conversion from returned value of type 'RWByteAddressBuffer' to function return type 'bool'}}
}

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    Fail_Bool();
}

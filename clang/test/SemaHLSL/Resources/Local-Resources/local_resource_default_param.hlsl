// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// Test that a resource parameter with a default value followed by a
// parameter without a default is rejected (standard C++ rule).
//
// DXC: also fails sema with the same message:
//   "missing default argument on parameter 'offset'"

RWByteAddressBuffer gBuf0 : register(u0);

uint Fail_DefaultParam(RWByteAddressBuffer buf = gBuf0, uint offset)
    // expected-error@-1 {{missing default argument on parameter 'offset'}}
{
    buf.Store(offset, 42);
    return 42;
}

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    Fail_DefaultParam(gBuf0, tid.x * 4);
}

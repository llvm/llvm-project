// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// Test that a compile-time out-of-bounds index into a global resource
// array produces a warning in clang but still compiles.
//
// DXC: fails sema with a hard error:
//   "array index 5 is out of bounds"

// expected-note@+1{{array 'gBufArray' declared here}}
RWByteAddressBuffer gBufArray[4] : register(u0);

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    RWByteAddressBuffer buf = gBufArray[5];
    // expected-warning@-1 {{array index 5 is past the end of the array (that has type 'RWByteAddressBuffer[4]')}}
    
    buf.Store(tid.x * 4, 42);
}

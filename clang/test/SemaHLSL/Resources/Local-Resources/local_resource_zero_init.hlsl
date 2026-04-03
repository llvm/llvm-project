// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// Test that brace (zero) initialization of a local resource is rejected
// because the empty initializer list has zero elements but the resource
// type expects one. This is distinct from default initialization (no
// initializer) which is accepted.
//
// DXC: also fails sema with:
//   "too few elements in vector initialization (expected 1 element, have 0)"

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    RWByteAddressBuffer buf = {};
    // expected-error@-1 {{too few initializers in list for type 'RWByteAddressBuffer' (expected 1 but found 0)}}
    buf.Store(tid.x * 4, 42);
}

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - -verify

// DXC passes sema but fails during codegen (DxilCondenseResources) with:
//   "local resource not guaranteed to map to unique global resource"
// Clang passes both sema and codegen, emitting a -Whlsl-explicit-binding
// warning instead.

RWByteAddressBuffer gBuf0 : register(u0);
RWByteAddressBuffer gBuf1 : register(u1);

uint Pass_IfAlias(bool cond, uint idx) {
    RWByteAddressBuffer buf;
    // DXC: error after sema: local resource not guaranteed to map to unique global resource.
    // expected-warning@+1{{assignment of 'cond ? gBuf0 : gBuf1' to local resource 'buf' is not to the same unique global resource}}
    buf = cond ? gBuf0 : gBuf1;
    buf.Store(idx * 4, 14);

    return 14;
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x + tid.y * 8;
    Pass_IfAlias(idx < 32, idx);
}

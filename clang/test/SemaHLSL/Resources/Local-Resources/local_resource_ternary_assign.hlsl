// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - -verify

// Test that ternary resource assignment post-declaration triggers a
// warning in clang but still produces valid IR.
//
// DXC: passes sema but fails codegen (DxilCondenseResources) with:
//   "local resource not guaranteed to map to unique global resource."

RWByteAddressBuffer gBuf0 : register(u0);
RWByteAddressBuffer gBuf1 : register(u1);

uint Pass_TernaryInit(bool cond, uint idx) {
    // DXC emits this warning: local resource not guaranteed to map to unique global resource.
    // expected-warning@+1{{assignment of 'cond ? gBuf0 : gBuf1' to local resource 'buf' is not to the same unique global resource}}
    RWByteAddressBuffer buf = cond ? gBuf0 : gBuf1;
    buf.Store(idx * 4, 2);

    return 2;
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x + tid.y * 8;
    Pass_TernaryInit(idx < 32, idx);    
}

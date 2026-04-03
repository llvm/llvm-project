// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute -emit-llvm -disable-llvm-passes 2>&1 -o - %s | llvm-cxxfilt | FileCheck %s

// DXC passes sema but fails during codegen (DxilCondenseResources) with:
//   "local resource not guaranteed to map to unique global resource"
// Clang passes both sema and codegen, emitting a -Whlsl-explicit-binding
// warning instead.

RWByteAddressBuffer gBuf0 : register(u0);
RWByteAddressBuffer gBuf1 : register(u1);

// CHECK: warning: assignment of 'gBuf1' to local resource 'buf' is not to the same unique global resource
RWByteAddressBuffer Pass_MultipleReturns(bool cond, uint idx) {
    RWByteAddressBuffer buf = gBuf0;
    if (cond) {
        buf.Store(idx * 4, 1);
        return buf;
    }
    // DXC: error after sema: local resource not guaranteed to map to unique global resource.
    buf = gBuf1;
    buf.Store(idx * 4, 2);
    return buf;
}

// CHECK: define hidden void @Pass_MultipleReturns(bool, unsigned int)(
// CHECK: define void @main()

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x + tid.y * 8;
    RWByteAddressBuffer result = Pass_MultipleReturns(idx < 32, idx);
    result.Store(0, idx);
}

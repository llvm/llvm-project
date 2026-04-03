// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute -emit-llvm -disable-llvm-passes 2>&1 -o - %s | llvm-cxxfilt | FileCheck %s

// DXC passes sema but fails during codegen (DxilCondenseResources) with:
//   "local resource not guaranteed to map to unique global resource"
// Clang passes both sema and codegen, emitting a -Whlsl-explicit-binding
// warning instead.

RWByteAddressBuffer gBuf0 : register(u0);
RWByteAddressBuffer gBuf2 : register(u2);

uint Pass_PhiMerge(bool cond, uint idx) {
    RWByteAddressBuffer buf;
    // DXC: error after sema: local resource not guaranteed to map to unique global resource.
    // CHECK: warning: assignment of 'cond ? gBuf0 : gBuf2' to local resource 'buf' is not to the same unique global resource
    buf = cond ? gBuf0 : gBuf2;
    buf.Store(idx * 4, 18);

    return 18;
}

// CHECK: define hidden noundef i32 @Pass_PhiMerge(bool, unsigned int)(
// CHECK: define void @main()

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x + tid.y * 8;
    Pass_PhiMerge(idx < 32, idx);
}

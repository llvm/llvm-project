// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute -emit-llvm -disable-llvm-passes 2>&1 -o - %s | llvm-cxxfilt | FileCheck %s

// DXC passes sema but fails during codegen (DxilCondenseResources) with:
//   "local resource not guaranteed to map to unique global resource"
// Clang passes both sema and codegen, emitting a -Whlsl-explicit-binding
// warning instead.

RWByteAddressBuffer gBuf0 : register(u0);
RWByteAddressBuffer gBuf1 : register(u1);

// CHECK: warning: assignment of 'gBuf1' to local resource 'buf' is not to the same unique global resource
[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    RWByteAddressBuffer buf = gBuf0;

    for (uint i = 0; i < 4; i++) {
        if (i == 2) {
            // DXC: error after sema: local resource not guaranteed to map to unique global resource.
            buf = gBuf1;
            continue;
        }
        buf.Store(i * 4, i);
    }

    buf.Store(tid.x * 4, 99);
}

// CHECK: define void @main()
// CHECK-NOT: error:

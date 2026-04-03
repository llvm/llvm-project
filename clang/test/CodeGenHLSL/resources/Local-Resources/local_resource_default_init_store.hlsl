// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - 2>&1 | llvm-cxxfilt | FileCheck %s

// Test that using a default-initialized (unbound) local resource produces
// valid IR in clang. No warnings or errors from clang.
//
// DXC: passes sema but fails codegen with:
//   "local resource not guaranteed to map to unique global resource."

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    RWByteAddressBuffer buf;
    buf.Store(tid.x * 4, 42);
}

// CHECK-NOT: error:
// CHECK-NOT: warning:
// CHECK: define {{.*}} @main(

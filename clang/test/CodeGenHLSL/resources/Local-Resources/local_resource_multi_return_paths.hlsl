// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - 2>&1 | llvm-cxxfilt | FileCheck %s

// Test that a function with multiple return paths returning different
// global resources, used to initialize a local, produces valid IR in clang.
//
// Clang accepts silently (no -Whlsl-explicit-binding warning is emitted
// because the initializer expression is a function call, not a direct
// global reference).
//
// DXC: passes sema but fails codegen with:
//   "local resource not guaranteed to map to unique global resource."
// (the error is reported at the first return statement inside Pick.)

RWByteAddressBuffer g0 : register(u0);
RWByteAddressBuffer g1 : register(u1);

RWByteAddressBuffer Pick(bool c)
{
    if (c) return g0;
    return g1;
}

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    RWByteAddressBuffer buf = Pick(tid.x > 0);
    buf.Store(tid.x * 4, 42);
}

// CHECK-NOT: error:
// CHECK: define {{.*}} @main(

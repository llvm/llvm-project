// REQUIRES: asserts
// RUN: not --crash %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.6-compute %s -emit-obj -O1 -o %t 2>&1 | FileCheck %s

// The DirectX backend turns an ambiguous local-resource access into an error.
// local_resource_branched_reassign_ambiguous.hlsl covers the Sema warning for
// this same construct; this test covers the backend error it lowers to.
// The condition is runtime-valued so the optimizer cannot fold the ambiguity
// away; this is deliberately the only CodeGenHLSL ambiguity test.
//
// TODO: DXILResourceAccess diagnoses the access but leaves it unlowered, so
// DXILOpLowering asserts in cleanupHandleCasts. Hence `not --crash` plus the
// asserts requirement (a release build would be UB rather than a clean crash).
// Drop both once the backend bails out cleanly after the error.

RWByteAddressBuffer GBuf1 : register(u1);
RWByteAddressBuffer GBuf2 : register(u2);

[numthreads(8,8,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
    RWByteAddressBuffer Buf = GBuf1;
    if (Tid.x & 1)
        Buf = GBuf2;
    Buf.Store(Tid.x * 4, 32);
}

// The store cannot be attributed to a single binding, so the backend reports
// both candidate handles (u1 and u2) before erroring out.
// CHECK-DAG: note: Uses resource handle:{{.*}}handlefrombinding{{.*}}(i32 0, i32 1, i32 1, i32 0
// CHECK-DAG: note: Uses resource handle:{{.*}}handlefrombinding{{.*}}(i32 0, i32 2, i32 1, i32 0
// CHECK: error: Resource access is not guaranteed to map to a unique global resource

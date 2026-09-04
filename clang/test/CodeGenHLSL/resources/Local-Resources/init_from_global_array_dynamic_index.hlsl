// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.6-compute %s -emit-llvm -O1 -o - | FileCheck %s

RWByteAddressBuffer GBufArray[4] : register(u0);

// CHECK-LABEL: define {{.*}}@main(
[numthreads(1,1,1)]
void main(uint3 Tid : SV_DispatchThreadID) {
// The index is not statically known, so it must survive as a runtime value and
// be threaded into the index operand of the binding rather than folded away.
// The binding covers the whole array: register u0, space 0, range 4.
// CHECK: %[[TID:.*]] = tail call i32 @llvm.dx.thread.id(i32 0)
// CHECK: %[[AND:.*]] = and i32 %[[TID]], 3
// CHECK: call target("dx.RawBuffer", i8, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_1_0t(i32 0, i32 0, i32 4, i32 %[[AND]],
    RWByteAddressBuffer Buf = GBufArray[Tid.x & 3];
    Buf.Store(0, 42);
}

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,DXCHECK

// SPIR-V codegen for RasterizerOrderedByteAddressBuffer is not implemented
// yet (asserts in clang/lib/CodeGen/Targets/SPIR.cpp on
// `!ResAttrs.IsROV && "Rasterizer order views not implemented for SPIR-V yet"`).
// Add a `spirv-pc-vulkan1.3-library` RUN line here when SPIR-V ROV support
// lands.

RasterizerOrderedByteAddressBuffer ROVB : register(u1);

// CHECK-LABEL: define void @{{.*}}test_rovb_float
// DXCHECK: %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 1), ptr {{.*}}
// DXCHECK: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_1t.i32(target("dx.RawBuffer", i8, 1, 1) %[[HANDLE]], i32 %{{.*}})
// DXCHECK: %[[CMP:.*]] = bitcast float %{{.*}} to i32
// DXCHECK-NEXT: %[[VAL:.*]] = bitcast float %{{.*}} to i32
// DXCHECK-NEXT: cmpxchg ptr %[[PTR]], i32 %[[CMP]], i32 %[[VAL]] syncscope("device") monotonic monotonic
export void test_rovb_float(uint off, float cmp, float v) {
  ROVB.InterlockedCompareStoreFloatBitwise(off, cmp, v);
}

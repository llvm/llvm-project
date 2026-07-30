// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,DXCHECK

// SPIR-V codegen for RasterizerOrderedByteAddressBuffer is not implemented
// yet (asserts in clang/lib/CodeGen/Targets/SPIR.cpp on
// `!ResAttrs.IsROV && "Rasterizer order views not implemented for SPIR-V yet"`).
// Add a `spirv-pc-vulkan1.3-library` RUN line here when SPIR-V ROV support
// lands.

RasterizerOrderedByteAddressBuffer ROVB : register(u1);

// CHECK-LABEL: define void @{{.*}}test_rovb_int_2arg
// DXCHECK: %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 1), ptr {{.*}}
// DXCHECK: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_1t.i32(target("dx.RawBuffer", i8, 1, 1) %[[HANDLE]], i32 %{{.*}})
// DXCHECK: atomicrmw add ptr %[[PTR]], i32 %{{.*}} monotonic
export void test_rovb_int_2arg(uint off, int v) {
  ROVB.InterlockedAdd(off, v);
}

// CHECK-LABEL: define void @{{.*}}test_rovb_uint_3arg
// DXCHECK: %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 1), ptr {{.*}}
// DXCHECK: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_1t.i32(target("dx.RawBuffer", i8, 1, 1) %[[HANDLE]], i32 %{{.*}})
// DXCHECK: %[[R:.*]] = atomicrmw add ptr %[[PTR]], i32 %{{.*}} monotonic
// DXCHECK: store i32 %[[R]], ptr {{.*}}
export void test_rovb_uint_3arg(uint off, uint v, out uint orig) {
  ROVB.InterlockedAdd(off, v, orig);
}

// CHECK-LABEL: define void @{{.*}}test_rovb_int64_2arg
// DXCHECK: %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 1), ptr {{.*}}
// DXCHECK: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_1t.i32(target("dx.RawBuffer", i8, 1, 1) %[[HANDLE]], i32 %{{.*}})
// DXCHECK: atomicrmw add ptr %[[PTR]], i64 %{{.*}} monotonic
export void test_rovb_int64_2arg(uint off, int64_t v) {
  ROVB.InterlockedAdd64(off, v);
}

// CHECK-LABEL: define void @{{.*}}test_rovb_uint64_3arg
// DXCHECK: %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 1), ptr {{.*}}
// DXCHECK: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_1t.i32(target("dx.RawBuffer", i8, 1, 1) %[[HANDLE]], i32 %{{.*}})
// DXCHECK: %[[R:.*]] = atomicrmw add ptr %[[PTR]], i64 %{{.*}} monotonic
// DXCHECK: store i64 %[[R]], ptr {{.*}}
export void test_rovb_uint64_3arg(uint off, uint64_t v, out uint64_t orig) {
  ROVB.InterlockedAdd64(off, v, orig);
}
